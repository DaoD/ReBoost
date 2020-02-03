import tensorflow as tf

class BatchedInput():
    def __init__(
            self,
            initializer,
            encoder_inputs,
            decoder_inputs,
            decoder_targets,
            src_length,
            tgt_length,
            retrieval_inputs=None,
            retrieval_length=None,
            retrieval_score=None,
            target_retrieval=None,
            target_retrieval_mask=None,
    ):
        self.initializer = initializer
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets
        self.src_length = src_length
        self.tgt_length = tgt_length
        self.retrieval_inputs = retrieval_inputs
        self.retrieval_length = retrieval_length
        self.target_retrieval = target_retrieval
        self.target_retrieval_mask = target_retrieval_mask
        self.retrieval_score = retrieval_score

def get_iterator(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        batch_size,
        sos,
        eos,
        source_reverse,
        src_max_len=None,
        tgt_max_len=None,
        num_threads=4,
        output_buffer_size=None,
        skip_count=None,
        retrieval_dataset=None,
        retrieval_length_dataset=None,
        retrieval_score_dataset=None,
        tw_unk=None,
        topic_vocab_table=None,
):
    if not output_buffer_size: output_buffer_size = batch_size * 1000
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)
    retrieval_unk_id = tf.cast(topic_vocab_table.lookup(tf.constant(tw_unk)), tf.int32)
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, retrieval_dataset, retrieval_length_dataset, retrieval_score_dataset))
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)
    src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size)
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, retrieval, retrieval_len, retrieval_score: (src, tgt, tf.string_split([retrieval], delimiter="\t").values, tf.string_to_number(tf.string_split([retrieval_len]).values, tf.int32), tf.string_to_number(tf.string_split([retrieval_score], delimiter="\t").values, tf.float32)))
    src_tgt_dataset = src_tgt_dataset.filter(lambda src, tgt, retrieval, retrieval_len, retrieval_score: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
    # This is used for STC dataset. When applying to DailyDialog, you should only use five retrieval results.
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, retrieval, retrieval_len, retrieval_score: (tf.string_split([src]).values, tf.string_split([
        tgt]).values, [tf.string_split([retrieval[0]]).values,
                       tf.string_split([retrieval[1]]).values,
                       tf.string_split([retrieval[2]]).values,
                       tf.string_split([retrieval[3]]).values,
                       tf.string_split([retrieval[4]]).values,
                       tf.string_split([retrieval[5]]).values,
                       tf.string_split([retrieval[6]]).values,
                       tf.string_split([retrieval[7]]).values,
                       tf.string_split([retrieval[8]]).values,
                       tf.string_split([retrieval[9]]).values],
                       retrieval_len, retrieval_score), num_parallel_calls=num_threads)
    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, retrieval, retrieval_len, retrieval_score: (src[:src_max_len], tgt, retrieval, retrieval_len, retrieval_score), num_parallel_calls=num_threads)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, retrieval, retrieval_len, retrieval_score: (src, tgt[:tgt_max_len], retrieval, retrieval_len, retrieval_score), num_parallel_calls=num_threads)
    if source_reverse:
        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, retrieval, retrieval_len, retrieval_score: (tf.reverse(src, axis=[0]), tgt, retrieval, retrieval_len, retrieval_score), num_parallel_calls=num_threads)
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, retrieval, retrieval_len, retrieval_score: (tf.cast(src_vocab_table.lookup(src), tf.int32), tf.cast(tgt_vocab_table.lookup(tgt), tf.int32), tf.cast(topic_vocab_table.lookup(tgt), tf.int32), tf.cast(tgt_vocab_table.lookup(retrieval), tf.int32), retrieval_len, retrieval_score), num_parallel_calls=num_threads)
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, tgt_rw, retrieval, retrieval_len, retrieval_score: (src, tgt, tgt_rw, tf.cast(tf.not_equal(tgt_rw, retrieval_unk_id), tf.float32), retrieval, retrieval_len, retrieval_score), num_parallel_calls=num_threads)
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, tgt_rw, tgt_rw_mask, retrieval, retrieval_len, retrieval_score: (src, tf.concat(([tgt_sos_id], tgt), 0), tf.concat((tgt, [tgt_eos_id]), 0), tf.concat((tgt_rw, [tgt_eos_id]), 0), tf.concat((tgt_rw_mask, [0]), 0), retrieval, retrieval_len, retrieval_score), num_parallel_calls=num_threads)
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt_in, tgt_out, tgt_rw, tgt_rw_mask, retrieval, retrieval_len, retrieval_score: (src, tgt_in, tgt_out, tgt_rw, tgt_rw_mask, retrieval, tf.size(src), tf.size(tgt_in), retrieval_len, retrieval_score), num_parallel_calls=num_threads)
    batched_dataset = src_tgt_dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None])), padding_values=(src_eos_id, tgt_eos_id, tgt_eos_id, retrieval_unk_id, 0.0, tgt_eos_id, 0, 0, 0, 0.0))
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_input_ids, tgt_output_ids, tgt_rw_ids, tgt_rw_mask, retrieval_ids, src_seq_len, tgt_seq_len, retrieval_len, retrieval_score) = (batched_iter.get_next())
    return BatchedInput(batched_iter.initializer, encoder_inputs=src_ids, decoder_inputs=tgt_input_ids, decoder_targets=tgt_output_ids, src_length=src_seq_len, tgt_length=tgt_seq_len, retrieval_inputs=retrieval_ids, retrieval_length=retrieval_len, retrieval_score=retrieval_score, target_retrieval=tgt_rw_ids, target_retrieval_mask=tgt_rw_mask)

def get_infer_iterator(
        src_dataset,
        src_vocab_table,
        batch_size,
        source_reverse,
        eos,
        src_max_len=None,
        retrieval_dataset=None,
        retrieval_length_dataset=None,
        retrieval_score_dataset=None,
        tgt_vocab_table=None,
):
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)
    src_dataset = tf.data.Dataset.zip((src_dataset, retrieval_dataset, retrieval_length_dataset, retrieval_score_dataset))
    src_dataset = src_dataset.map(lambda src, retrieval, retrieval_len, retrieval_score: (src, tf.string_split([retrieval], delimiter="\t").values, tf.string_to_number(tf.string_split([retrieval_len]).values, tf.int32), tf.string_to_number(tf.string_split([retrieval_score], delimiter="\t").values, tf.float32)))
    src_dataset = src_dataset.map(lambda src, retrieval, retrieval_len, retrieval_score: (tf.string_split([src]).values, [tf.string_split([retrieval[0]]).values, tf.string_split([retrieval[1]]).values, tf.string_split([retrieval[2]]).values, tf.string_split([retrieval[3]]).values, tf.string_split([retrieval[4]]).values, tf.string_split([retrieval[5]]).values, tf.string_split([retrieval[6]]).values, tf.string_split([retrieval[7]]).values, tf.string_split([retrieval[8]]).values, tf.string_split([retrieval[9]]).values], retrieval_len, retrieval_score))
    if src_max_len:
        src_dataset = src_dataset.map(lambda src, retrieval, retrieval_len, retrieval_score: (src[:src_max_len], retrieval, retrieval_len, retrieval_score))
    src_dataset = src_dataset.map(lambda src, retrieval, retrieval_len, retrieval_score: (tf.cast(src_vocab_table.lookup(src), tf.int32), tf.cast(tgt_vocab_table.lookup(retrieval), tf.int32), retrieval_len, retrieval_score))
    if source_reverse:
        src_dataset = src_dataset.map(lambda src, retrieval, retrieval_len, retrieval_score: (tf.reverse(src, axis=[0]), retrieval, retrieval_len, retrieval_score))
    src_dataset = src_dataset.map(lambda src, retrieval, retrieval_len, retrieval_score: (src, retrieval, tf.size(src), retrieval_len, retrieval_score))
    batched_dataset = src_dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None])), padding_values=(src_eos_id, tgt_eos_id, 0, 0, 0.0))
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, retrieval_ids, src_seq_len, retrieval_len, retrieval_score) = batched_iter.get_next()

    return BatchedInput(batched_iter.initializer, encoder_inputs=src_ids, decoder_inputs=None, decoder_targets=None, src_length=src_seq_len,
                        tgt_length=None, retrieval_inputs=retrieval_ids, retrieval_length=retrieval_len, retrieval_score=retrieval_score)
