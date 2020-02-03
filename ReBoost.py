import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from modelConfig import FLAGS
from myGreedyEmbeddingHelper import MyGreedyEmbeddingHelper
from myBasicDecoder import BasicDecoder
from myBeamSearchDecoder import BeamSearchDecoder
from ModifiedAttention import ModifiedAttention

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

class MyModel():
    def __init__(
            self,
            data_iterator,
            src_vocab_table,
            tgt_vocab_table,
            mode,
            overlap_matrix=None,
            reverse_target_vocab_table=None,
            scope=None,
    ):
        self.mode = mode
        self.data_iterator = data_iterator
        self.batch_size = tf.size(self.data_iterator.src_length)
        self.src_vocab_size = FLAGS.src_vocab_size
        self.tgt_vocab_size = FLAGS.tgt_vocab_size
        self.time_major = FLAGS.time_major
        self.src_vocab_table = src_vocab_table
        self.tgt_vocab_table = tgt_vocab_table
        self.retrieval_embedding_size = FLAGS.retrieval_embedding_size
        self.retrieval_num = FLAGS.retrieval_num
        self.overlap_matrix = tf.constant(overlap_matrix)
        self.retrieval_keywords_vocab_size = FLAGS.retrieval_keywords_vocab_size

        initializer = tf.random_uniform_initializer(-FLAGS.init_weight, FLAGS.init_weight)
        tf.get_variable_scope().set_initializer(initializer)
        res = self.build_graph(FLAGS.hidden_size, FLAGS.embedding_size, scope)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss_all = res[1]
            self.train_loss = res[2]
            self.word_count = tf.reduce_sum(data_iterator.src_length) + tf.reduce_sum(data_iterator.tgt_length)
            self.final_context_state = res[3]
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss_all = res[1]
            self.eval_loss = res[2]
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_logits, _, _, self.final_context_state, self.sample_id, self.alpha = res
            self.sample_words = reverse_target_vocab_table.lookup(tf.to_int64(self.sample_id))

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.predict_count = tf.reduce_sum(data_iterator.tgt_length)

        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()

        self.learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * FLAGS.decay_factor)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            if FLAGS.optimizer == "sgd":
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif FLAGS.optimizer == "adam":
                opt = tf.train.AdamOptimizer(self.learning_rate)
            elif FLAGS.optimizer == "adadelta":
                opt = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif FLAGS.optimizer == "adagrad":
                opt = tf.train.AdagradOptimizer(self.learning_rate)
            tf.summary.scalar("lr", self.learning_rate)
            gradients = tf.gradients(self.train_loss, params, colocate_gradients_with_ops=FLAGS.colocate_gradients_with_ops)
            # clip gradient
            clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
            gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
            gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))
            self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            # summary
            self.train_summary = tf.summary.merge(
                [tf.summary.scalar("lr", self.learning_rate), tf.summary.scalar("train_loss", self.train_loss)] + gradient_norm_summary)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    def embedding_attention_seq2seq(
            self,
            hidden_size,
            embedding_size,
            scope,
            dtype
    ):
        beam_width = FLAGS.beam_width
        batch_size = self.batch_size
        length_penalty_weight = FLAGS.length_penalty_weight
        src_vocab_size = self.src_vocab_size
        tgt_vocab_size = self.tgt_vocab_size
        time_major = self.time_major
        data_iterator = self.data_iterator
        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(SOS)), tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(EOS)), tf.int32)
        encoder_inputs = data_iterator.encoder_inputs
        retrieval_inputs = data_iterator.retrieval_inputs
        retrieval_score = data_iterator.retrieval_score

        if self.time_major:
            encoder_inputs = tf.transpose(encoder_inputs)
            retrieval_inputs = tf.transpose(retrieval_inputs)

        with tf.variable_scope(scope or "build_network"):
            with tf.variable_scope("decoder/output_projection"):
                output_layer = layers_core.Dense(tgt_vocab_size, use_bias=False, name="output_projection")
                output_keywords_layer = layers_core.Dense(self.retrieval_keywords_vocab_size, use_bias=False, name="keywords_output_projection")
                attention_layer = layers_core.Dense(hidden_size, use_bias=False, name="attention_projection")

        with tf.variable_scope(scope or "embedding_attention_seq2seq", dtype=dtype):
            self.embedding_encoder = tf.get_variable("embedding_encoder", [src_vocab_size, embedding_size], dtype)
            self.embedding_decoder = tf.get_variable("embedding_decoder", [tgt_vocab_size, embedding_size], dtype)

            with tf.variable_scope("encoder"):
                encoder_emb_input = tf.nn.embedding_lookup(self.embedding_encoder, encoder_inputs)
                fw_encoder_cell = tf.contrib.rnn.GRUCell(hidden_size)
                bw_encoder_cell = tf.contrib.rnn.GRUCell(hidden_size)
                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_encoder_cell,
                    bw_encoder_cell,
                    encoder_emb_input,
                    dtype=dtype,
                    sequence_length=data_iterator.src_length,
                    time_major=time_major)
                encoder_outputs = tf.concat(bi_outputs, -1)
                encoder_state = tf.concat([bi_state[0], bi_state[1]], axis=1)

            alpha = None
            with tf.variable_scope("retrieval_encoder"):
                if time_major:
                    retrieval_inputs = tf.reshape(retrieval_inputs, [-1, self.retrieval_num * self.batch_size])
                else:
                    retrieval_inputs = tf.reshape(retrieval_inputs, [self.batch_size * self.retrieval_num, -1])
                retrieval_emb_inputs = tf.nn.embedding_lookup(self.embedding_decoder, retrieval_inputs)
                retrieval_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.retrieval_embedding_size), input_keep_prob=0.5)
                retrieval_length = tf.reshape(data_iterator.retrieval_length, [-1])
                retrieval_outputs, _ = tf.nn.dynamic_rnn(retrieval_cell, retrieval_emb_inputs, dtype=dtype,  sequence_length=retrieval_length, time_major=time_major)
                with tf.variable_scope("retrieval_words_attention"):
                    u_context = tf.get_variable("u_context", [2 * hidden_size], dtype=dtype)
                    if time_major:
                        # shape: [batch_size, max_time, output_size]
                        retrieval_outputs = tf.transpose(retrieval_outputs, [1, 0, 2])
                    h = tf.layers.dense(retrieval_outputs, 2 * hidden_size, activation=tf.nn.tanh)
                    alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
                    retrieval_memory = tf.reduce_sum(tf.multiply(retrieval_outputs, alpha), axis=1)
                    if time_major:
                        retrieval_memory = tf.reshape(retrieval_memory, [self.retrieval_num, self.batch_size, self.retrieval_embedding_size])
                        retrieval_memory = tf.transpose(retrieval_memory, [1, 0, 2])
                    else:
                        retrieval_memory = tf.reshape(retrieval_memory, [self.batch_size, self.retrieval_num, self.retrieval_embedding_size])

            with tf.variable_scope("decoder") as decoder_scope:
                decoding_length_factor = 2.0
                max_encoder_length = tf.reduce_max(data_iterator.src_length)
                maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * decoding_length_factor))
                if time_major:
                    memory = tf.transpose(encoder_outputs, [1, 0, 2])
                else:
                    memory = encoder_outputs

                src_length = data_iterator.src_length
                if self.mode == tf.contrib.learn.ModeKeys.INFER and beam_width > 0:
                    memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
                    src_length = tf.contrib.seq2seq.tile_batch(data_iterator.src_length, multiplier=beam_width)
                    encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
                    retrieval_memory = tf.contrib.seq2seq.tile_batch(retrieval_memory, multiplier=beam_width)
                    retrieval_score = tf.contrib.seq2seq.tile_batch(retrieval_score, multiplier=beam_width)
                    batch_size = batch_size * beam_width

                message_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(hidden_size, memory, memory_sequence_length=src_length)  # for small test
                retrieval_attention_mechanism = ModifiedAttention(hidden_size, retrieval_memory, memory_sequence_length=None, retrieval_score=retrieval_score)
                attention_mechanism = [message_attention_mechanism, retrieval_attention_mechanism]

                decoder_cell = tf.contrib.rnn.GRUCell(hidden_size)
                alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and beam_width == 0)  # or (self.mode == tf.contrib.learn.ModeKeys.TRAIN)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=[hidden_size, self.retrieval_embedding_size], alignment_history=alignment_history, output_attention=False, name="attention")

                decoder_initial_state = decoder_cell.zero_state(batch_size, dtype).clone(cell_state=tf.layers.dense(encoder_state, hidden_size))

                if self.mode != tf.contrib.learn.ModeKeys.INFER:
                    decoder_inputs = data_iterator.decoder_inputs
                    if time_major:
                        decoder_inputs = tf.transpose(decoder_inputs)
                    decoder_emb_input = tf.nn.embedding_lookup(self.embedding_decoder, decoder_inputs)
                    helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_input, data_iterator.tgt_length, time_major=time_major)

                    my_decoder = BasicDecoder(decoder_cell, helper, decoder_initial_state, overlap_matrix=self.overlap_matrix, attention_output_layer=attention_layer)
                    decoder_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, output_time_major=time_major, swap_memory=True, scope=decoder_scope)
                    sample_id = decoder_outputs.sample_id
                    logits = output_layer(decoder_outputs.rnn_output)
                    keywords_logits = output_keywords_layer(decoder_outputs.weighted_sum)
                else:
                    start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                    end_token = tgt_eos_id
                    if beam_width > 0:
                        my_decoder = BeamSearchDecoder(cell=decoder_cell, embedding=self.embedding_decoder, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state, beam_width=beam_width, output_layer=output_layer,  length_penalty_weight=length_penalty_weight, output_layer_2=output_keywords_layer, attention_output_layer=attention_layer, overlap_matrix=self.overlap_matrix)
                    else:
                        helper = MyGreedyEmbeddingHelper(self.embedding_decoder, start_tokens, end_token, tgt_vocab_size)
                        my_decoder = BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=output_layer,  overlap_matrix=self.overlap_matrix, output_layer_2=output_keywords_layer, attention_output_layer=attention_layer)
                    decoder_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, maximum_iterations=maximum_iterations, 
                        output_time_major=time_major, swap_memory=True, scope=decoder_scope)
                    if beam_width > 0:
                        logits = tf.no_op()
                        keywords_logits = tf.no_op()
                        sample_id = decoder_outputs.predicted_ids
                    else:
                        logits = decoder_outputs.rnn_output
                        keywords_logits = decoder_outputs.weighted_sum
                        sample_id = decoder_outputs.sample_id
        return logits, keywords_logits, sample_id, final_context_state, alpha

    def build_graph(
            self,
            hidden_size,
            embedding_size,
            scope
    ):
        dtype = tf.float32
        data_iterator = self.data_iterator
        batch_size = self.batch_size
        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
            logits, keywords_logits, sample_id, final_context_state, alpha = self.embedding_attention_seq2seq(hidden_size, embedding_size, scope, dtype)
            time_major = self.time_major
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                target_output = data_iterator.decoder_targets
                target_retrieval = data_iterator.target_retrieval
                target_retrieval_mask = data_iterator.target_retrieval_mask
                if time_major:
                    target_output = tf.transpose(target_output)
                    target_retrieval = tf.transpose(target_retrieval)
                    target_retrieval_mask = tf.transpose(target_retrieval_mask)
                time_axis = 0 if time_major else 1
                max_time = target_output.shape[time_axis].value or tf.shape(target_output)[time_axis]
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
                target_weights = tf.sequence_mask(data_iterator.tgt_length, max_time, dtype=logits.dtype)
                if not FLAGS.no_bias:
                    crossent_keywords = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_retrieval, logits=keywords_logits)
                    crossent_keywords = crossent_keywords * target_retrieval_mask
                if time_major:
                    target_weights = tf.transpose(target_weights)
                    # crossent_keywords = tf.transpose(crossent_keywords)
                loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(batch_size)
                if not FLAGS.no_bias:
                    loss_all = (tf.reduce_sum(crossent * target_weights) + tf.reduce_sum(crossent_keywords * target_weights)) / tf.to_float(batch_size)
                else:
                    loss_all = loss
            else:
                loss = None
                loss_all = None
        return logits, loss_all, loss, final_context_state, sample_id, alpha

    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run(
            [self.update, self.train_loss, self.predict_count, self.train_summary, self.global_step, self.word_count, self.batch_size, self.final_context_state,
             self.train_loss_all])

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss, self.predict_count, self.batch_size, self.eval_loss_all])

    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run([self.infer_logits, self.sample_id, self.sample_words])

    def decode(self, sess):
        _, _, sample_words = self.infer(sess)
        if self.time_major:
            sample_words = sample_words.transpose()
        elif sample_words.ndim == 3:
            sample_words = sample_words.transpose([2, 0, 1])
        return sample_words
