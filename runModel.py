import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.python import debug as tf_debug
import os
import time
import random
import math
import re
import ReBoost
import iterator_utils
import evaluation_utils
import pickle
from modelConfig import FLAGS

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"

UNK_ID = 0
_START_VOCAB = [UNK, SOS, EOS]
_WORD_SPLIT = re.compile("[^，。？！\w]")
_DIGIT_RE = re.compile("\d")


def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, tokenizer=None, normalize_digits=True):
    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with tf.gfile.GFile(data_path, mode='r') as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                # line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub("0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with tf.gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")


def create_train_model(model_creator, scope=None):
    train_src_file = FLAGS.train_src_file
    train_tgt_file = FLAGS.train_tgt_file
    src_vocab_file = FLAGS.src_vocab_file
    tgt_vocab_file = FLAGS.tgt_vocab_file
    train_graph = tf.Graph()

    with train_graph.as_default():
        src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)
        train_src_dataset = tf.data.TextLineDataset(train_src_file)
        train_tgt_dataset = tf.data.TextLineDataset(train_tgt_file)
        train_skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)
        train_retrieval_dataset = tf.data.TextLineDataset(FLAGS.train_retrieval_file)
        train_retrieval_length_dataset = tf.data.TextLineDataset(FLAGS.train_retrieval_length_file)
        trian_retrieval_score_dataset = tf.data.TextLineDataset(FLAGS.train_retrieval_score_file)
        retrieval_vocab_file = FLAGS.retrieval_vocab_file
        retrieval_vocab_table = lookup_ops.index_table_from_file(retrieval_vocab_file, default_value=UNK_ID)
        train_iterator = iterator_utils.get_iterator(
            train_src_dataset,
            train_tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=FLAGS.batch_size,
            sos=SOS,
            eos=EOS,
            source_reverse=FLAGS.source_reverse,
            skip_count=train_skip_count_placeholder,
            retrieval_dataset=train_retrieval_dataset,
            retrieval_length_dataset=train_retrieval_length_dataset,
            retrieval_score_dataset=trian_retrieval_score_dataset,
            tw_unk=UNK,
            topic_vocab_table=retrieval_vocab_table
        )
        overlap_matrix = pickle.load(open(FLAGS.overlap_matrix, 'rb'))
        # overlap_matrix_idx = pickle.load(open(FLAGS.overlap_matrix_idx, 'rb'))
        # overlap_matrix_val = pickle.load(open(FLAGS.overlap_matrix_val, 'rb'))
        # overlap_matrix_shape = pickle.load(open(FLAGS.overlap_matrix_shape, 'rb'))
        train_model = model_creator(data_iterator=train_iterator, src_vocab_table=src_vocab_table, tgt_vocab_table=tgt_vocab_table,
                                    mode=tf.contrib.learn.ModeKeys.TRAIN, overlap_matrix=overlap_matrix, scope=scope)
    return train_graph, train_model, train_iterator, train_skip_count_placeholder


def create_eval_model(model_creator, scope=None):
    src_vocab_file = FLAGS.src_vocab_file
    tgt_vocab_file = FLAGS.tgt_vocab_file
    eval_graph = tf.Graph()

    with eval_graph.as_default():
        src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)
        eval_src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        eval_tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        eval_retrieval_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        eval_retrieval_length_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        eval_retrieval_score_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        eval_src_dataset = tf.data.TextLineDataset(eval_src_file_placeholder)
        eval_tgt_dataset = tf.data.TextLineDataset(eval_tgt_file_placeholder)
        eval_retrieval_dataset = tf.data.TextLineDataset(eval_retrieval_file_placeholder)
        eval_retrieval_length_dataset = tf.data.TextLineDataset(eval_retrieval_length_file_placeholder)
        eval_retrieval_score_dataset = tf.data.TextLineDataset(eval_retrieval_score_file_placeholder)
        retrieval_vocab_file = FLAGS.retrieval_vocab_file
        retrieval_vocab_table = lookup_ops.index_table_from_file(retrieval_vocab_file, default_value=UNK_ID)
        eval_iterator = iterator_utils.get_iterator(
            eval_src_dataset,
            eval_tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=FLAGS.batch_size,
            sos=SOS,
            eos=EOS,
            source_reverse=False,
            retrieval_dataset=eval_retrieval_dataset,
            retrieval_length_dataset=eval_retrieval_length_dataset,
            retrieval_score_dataset=eval_retrieval_score_dataset,
            tw_unk=UNK,
            topic_vocab_table=retrieval_vocab_table,
        )
        overlap_matrix = pickle.load(open(FLAGS.overlap_matrix, 'rb'))
        # overlap_matrix_idx = pickle.load(open(FLAGS.overlap_matrix_idx, 'rb'))
        # overlap_matrix_val = pickle.load(open(FLAGS.overlap_matrix_val, 'rb'))
        # overlap_matrix_shape = pickle.load(open(FLAGS.overlap_matrix_shape, 'rb'))
        eval_model = model_creator(data_iterator=eval_iterator, src_vocab_table=src_vocab_table, tgt_vocab_table=tgt_vocab_table,
                                   mode=tf.contrib.learn.ModeKeys.EVAL,
                                   overlap_matrix=overlap_matrix, scope=scope)
        return eval_graph, eval_model, eval_src_file_placeholder, eval_tgt_file_placeholder, eval_retrieval_file_placeholder, eval_retrieval_length_file_placeholder, \
               eval_retrieval_score_file_placeholder, eval_iterator


def create_infer_model(model_creator, scope=None):
    src_vocab_file = FLAGS.src_vocab_file
    tgt_vocab_file = FLAGS.tgt_vocab_file
    infer_graph = tf.Graph()

    with infer_graph.as_default():
        src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(tgt_vocab_file, default_value=UNK)
        infer_src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        infer_retrieval_file_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        infer_retrieval_length_file_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        infer_retrieval_score_file_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        infer_batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        infer_src_dataset = tf.data.Dataset.from_tensor_slices(infer_src_placeholder)
        infer_retrieval_dataset = tf.data.Dataset.from_tensor_slices(infer_retrieval_file_placeholder)
        infer_retrieval_length_dataset = tf.data.Dataset.from_tensor_slices(infer_retrieval_length_file_placeholder)
        infer_retrieval_score_dataset = tf.data.Dataset.from_tensor_slices(infer_retrieval_score_file_placeholder)
        infer_iterator = iterator_utils.get_infer_iterator(
            infer_src_dataset,
            src_vocab_table,
            batch_size=FLAGS.infer_batch_size,
            source_reverse=FLAGS.source_reverse,
            eos=EOS,
            retrieval_dataset=infer_retrieval_dataset,
            retrieval_length_dataset=infer_retrieval_length_dataset,
            retrieval_score_dataset=infer_retrieval_score_dataset,
            tgt_vocab_table=tgt_vocab_table,
        )
        overlap_matrix = pickle.load(open(FLAGS.overlap_matrix, 'rb'))
        # overlap_matrix_idx = pickle.load(open(FLAGS.overlap_matrix_idx, 'rb'))
        # overlap_matrix_val = pickle.load(open(FLAGS.overlap_matrix_val, 'rb'))
        # overlap_matrix_shape = pickle.load(open(FLAGS.overlap_matrix_shape, 'rb'))
        infer_model = model_creator(data_iterator=infer_iterator, src_vocab_table=src_vocab_table, tgt_vocab_table=tgt_vocab_table,
                                    mode=tf.contrib.learn.ModeKeys.INFER,
                                    overlap_matrix=overlap_matrix, reverse_target_vocab_table=reverse_tgt_vocab_table, scope=scope)
        return infer_graph, infer_model, infer_src_placeholder, infer_batch_size_placeholder, infer_retrieval_file_placeholder, infer_retrieval_length_file_placeholder, \
               infer_retrieval_score_file_placeholder, infer_iterator


def load_data(input_file):
    with tf.gfile.GFile(input_file, mode="rb") as f:
        data = f.read().splitlines()
    return data


def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    print("  loaded %s model parameters from %s, time %.2fs" % (name, ckpt, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print("  create %s model with fresh parameters, time %.2fs" % (name, time.time() - start_time))
    global_step = model.global_step.eval(session=session)
    return model, global_step

def compute_perplexity(model, sess, name):
    total_loss = 0
    total_predict_count = 0
    while True:
        try:
            loss, predict_count, batch_size, loss_all = model.eval(sess)
            total_loss += loss * batch_size
            total_predict_count += predict_count
        except tf.errors.OutOfRangeError:
            break
    try:
        perplexity = math.exp(total_loss / total_predict_count)
    except:
        perplexity = float("inf")
    print("%s: perplexity %.2f" % (name, perplexity))
    return perplexity

def decode_and_evaluate(name, model, sess, trans_file, ref_file, subword_option, beam_width, tgt_eos, decode=True):
    import collections
    if decode:
        print("  decoding to output %s." % trans_file)
        num_sentences = 0
        with tf.gfile.GFile(trans_file, mode="w") as trans_f:
            trans_f.write("")
            idx = 0
            while True:
                try:
                    model_outputs = model.decode(sess)
                    if beam_width > 0:
                        model_outputs = model_outputs[0]
                    num_sentences += len(model_outputs)
                    for sent_id in range(len(model_outputs)):
                        output = model_outputs[sent_id, :].tolist()
                        if tgt_eos and tgt_eos.encode("utf-8") in output:
                            output = output[:output.index(tgt_eos.encode("utf-8"))]
                        if not hasattr(output, "__len__") and not isinstance(output, collections.Iterable):
                            output = [output]
                        results = b" ".join(output)
                        trans_f.write(("%s\n" % (results.decode("utf-8"))))
                        idx += 1
                except tf.errors.OutOfRangeError:
                    print("  done, num sentences %d" % num_sentences)
                    break
    # Evaluation
    if ref_file and tf.gfile.Exists(trans_file):
        score = evaluation_utils.evaluate(ref_file, trans_file, subword_option=subword_option)
        print("bleu scores: %.1f" % score)

def train(scope=None, target_session=""):
    out_dir = FLAGS.out_dir
    num_train_steps = FLAGS.num_train_steps
    steps_per_stats = FLAGS.steps_per_stats
    steps_per_eval = 10 * steps_per_stats
    dev_src_file = FLAGS.dev_src_file
    dev_tgt_file = FLAGS.dev_tgt_file
    test_src_file = FLAGS.test_src_file
    test_tgt_file = FLAGS.test_tgt_file
    model_creator = ReBoost.MyModel
    dev_retrieval_file = FLAGS.dev_retrieval_file
    dev_retrieval_length_file = FLAGS.dev_retrieval_length_file
    dev_retrieval_score_file = FLAGS.dev_retrieval_score_file
    test_retrieval_file = FLAGS.test_retrieval_file
    test_retrieval_length_file = FLAGS.test_retrieval_length_file
    test_retrieval_score_file = FLAGS.test_retrieval_score_file
    train_graph, train_model, train_iterator, train_skip_count_placeholder = create_train_model(model_creator, scope)
    eval_graph, eval_model, eval_src_file_placeholder, eval_tgt_file_placeholder, eval_retrieval_file_placeholder, eval_retrieval_length_file_placeholder, eval_retrieval_score_file_placeholder, eval_iterator = create_eval_model(model_creator, scope)
    dev_eval_iterator_feed_dict = {eval_src_file_placeholder: dev_src_file, eval_tgt_file_placeholder: dev_tgt_file,
                                   eval_retrieval_file_placeholder: dev_retrieval_file,
                                   eval_retrieval_length_file_placeholder: dev_retrieval_length_file,
                                   eval_retrieval_score_file_placeholder: dev_retrieval_score_file}
    test_eval_iterator_feed_dict = {eval_src_file_placeholder: test_src_file, eval_tgt_file_placeholder: test_tgt_file,
                                    eval_retrieval_file_placeholder: test_retrieval_file,
                                    eval_retrieval_length_file_placeholder: test_retrieval_length_file,
                                    eval_retrieval_score_file_placeholder: test_retrieval_score_file}
    summary_name = "train_log"
    best_ppl, best_step = FLAGS.best_perplexity, 0
    model_dir = FLAGS.model_dir
    best_model_dir = FLAGS.best_model_dir
    avg_step_time = 0.0
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    train_sess = tf.Session(target=target_session, graph=train_graph)
    eval_sess = tf.Session(target=target_session, graph=eval_graph)
    # train_sess = tf_debug.LocalCLIDebugWrapperSession(train_sess)
    with train_graph.as_default():
        train_model, global_step = create_or_load_model(train_model, model_dir, train_sess, "train")
    summary_writer = tf.summary.FileWriter(os.path.join(out_dir, summary_name), train_graph)

    def get_perplexity(model_dir):
        with eval_graph.as_default():
            loaded_eval_model, global_step = create_or_load_model(eval_model, model_dir, eval_sess, "eval")
        eval_sess.run(eval_iterator.initializer, feed_dict=dev_eval_iterator_feed_dict)
        dev_ppl = compute_perplexity(loaded_eval_model, eval_sess, "dev")
        eval_sess.run(eval_iterator.initializer, feed_dict=test_eval_iterator_feed_dict)
        test_ppl = compute_perplexity(loaded_eval_model, eval_sess, "test")
        return dev_ppl, test_ppl

    step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
    checkpoint_total_count = 0.0
    speed, train_ppl = 0.0, 0.0
    epoch = 1
    start_train_time = time.time()
    print("# Start step %d, optimizer %s, lr %g, %s" % (global_step, FLAGS.optimizer, train_model.learning_rate.eval(session=train_sess), time.ctime()))
    skip_count = FLAGS.batch_size * FLAGS.epoch_step
    epoch_step = FLAGS.epoch_step
    print("# Init train iterator, skipping %d elements" % skip_count)
    train_sess.run(train_iterator.initializer, feed_dict={train_skip_count_placeholder: skip_count})
    while global_step < num_train_steps:
        start_time = time.time()
        try:
            step_result = train_model.train(train_sess)
            (_, step_loss, step_predict_count, step_summary, global_step, step_word_count, batch_size, final_context_state, step_loss_all) = step_result
            epoch_step += 1
        except tf.errors.OutOfRangeError:
            epoch_step = 0
            epoch += 1
            print("# Finished an epoch, step %d" % global_step)
            dev_ppl, test_ppl = get_perplexity(model_dir)
            train_sess.run(train_model.learning_rate_decay_op)
            print("# Do learning rate decay: %f, now the learning rate is %g" % (FLAGS.decay_factor, train_model.learning_rate.eval(session=train_sess)))
            train_sess.run(train_iterator.initializer, feed_dict={train_skip_count_placeholder: 0})
            continue
        summary_writer.add_summary(step_summary, global_step)
        step_time += (time.time() - start_time)
        checkpoint_loss += (step_loss * batch_size)
        checkpoint_predict_count += step_predict_count
        checkpoint_total_count += float(step_word_count)

        if global_step % steps_per_stats == 0:
            avg_step_time = step_time / steps_per_stats
            train_ppl = math.exp(checkpoint_loss / checkpoint_predict_count)
            speed = checkpoint_total_count / (1000 * step_time)
            print("# Epoch %d global step %d lr %g step-time %.2fs wps %.2fK train ppl %.2f" % (
            epoch, global_step, train_model.learning_rate.eval(session=train_sess), avg_step_time, speed, train_ppl))
            if math.isnan(train_ppl):
                break
            step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
            checkpoint_total_count = 0.0

        if global_step % steps_per_eval == 0:
            print("# Save eval, global step %d" % global_step)
            summary = tf.Summary(value=[tf.Summary.Value(tag="train_ppl", simple_value=train_ppl)])
            summary_writer.add_summary(summary, global_step)
            train_model.saver.save(train_sess, os.path.join(out_dir, "generate.ckpt"), global_step=global_step)
            dev_ppl, test_ppl = get_perplexity(model_dir)
            dev_summary = tf.Summary(value=[tf.Summary.Value(tag="dev_ppl", simple_value=dev_ppl)])
            test_summary = tf.Summary(value=[tf.Summary.Value(tag="test_ppl", simple_value=test_ppl)])
            summary_writer.add_summary(dev_summary, global_step)
            summary_writer.add_summary(test_summary, global_step)
            if dev_ppl < best_ppl:
                best_ppl = dev_ppl
                best_step = global_step
                eval_model.saver.save(eval_sess, os.path.join(best_model_dir, "generate.ckpt"))
                print("# Now the best ppl on dev: %.2f, obtained in global step %d" % (best_ppl, best_step))

    train_model.saver.save(train_sess, os.path.join(out_dir, "generate.ckpt"), global_step=global_step)
    dev_ppl, test_ppl = get_perplexity(model_dir)
    print("# Final, step %d lr %g step-time %.2f wps %.2fK ppl %.2f, dev ppl %.2f test ppl %.2f" % (
    global_step, train_model.learning_rate.eval(session=train_sess),
    avg_step_time, speed, train_ppl, dev_ppl, test_ppl))
    print("# Done training! %.2f" % (time.time() - start_train_time))
    print("# Start evaluating saved best models.")
    best_dev_ppl, best_test_ppl = get_perplexity(best_model_dir)
    print("# Best perplexity, step %d step-time %.2f wps %.2fK dev ppl %.2f test ppl %.2f" % (global_step, avg_step_time, speed, best_dev_ppl, best_test_ppl))
    summary_writer.close()
    return (dev_ppl, test_ppl, global_step)

def inference(scope=None):
    out_dir = FLAGS.out_dir
    infer_input_file = FLAGS.inference_input_file
    output_infer = FLAGS.inference_output_file
    infer_ref = FLAGS.inference_ref_file
    ckpt = FLAGS.ckpt
    if not ckpt:
        ckpt = tf.train.latest_checkpoint(out_dir)
    model_creator = ReBoost.MyModel
    retrieval_input_file = FLAGS.test_retrieval_file
    retrieval_length_input_file = FLAGS.test_retrieval_length_file
    retrieval_score_input_file = FLAGS.test_retrieval_score_file
    retrieval_data = load_data(retrieval_input_file)
    retrieval_length_data = load_data(retrieval_length_input_file)
    retrieval_score_data = load_data(retrieval_score_input_file)
    infer_graph, infer_model, infer_src_placeholder, infer_batch_size_placeholder, infer_retrieval_file_placeholder, infer_retrieval_length_file_placeholder, infer_retrieval_score_file_placeholder, infer_iterator = create_infer_model(model_creator, scope)
    infer_data = load_data(infer_input_file)
    config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})
    config.gpu_options.allow_growth = True
    print("# The length of input data:", len(retrieval_data))
    with tf.Session(graph=infer_graph) as sess:
        loaded_infer_model = load_model(infer_model, ckpt, sess, "infer")
        sess.run(infer_model.data_iterator.initializer, feed_dict={infer_src_placeholder: infer_data, infer_batch_size_placeholder: FLAGS.infer_batch_size, infer_retrieval_file_placeholder: retrieval_data, infer_retrieval_length_file_placeholder: retrieval_length_data, infer_retrieval_score_file_placeholder: retrieval_score_data})
        print("# Start decoding")
        decode_and_evaluate("infer", loaded_infer_model, sess, output_infer, infer_ref, subword_option=None, beam_width=FLAGS.beam_width, tgt_eos=EOS)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # create_vocabulary("./data/STC/src_vocab_file", "./data/STC/train/post.train.retrieval", 40000)
    # create_vocabulary("./data/STC/tgt_vocab_file", "./data/STC/train/cmnt.train.retrieval", 40000)
    train()
    # inference()

if __name__ == "__main__":
    main()
