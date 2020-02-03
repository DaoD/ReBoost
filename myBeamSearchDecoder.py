from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
import tensorflow as tf

class BeamSearchDecoderState(
    collections.namedtuple("BeamSearchDecoderState", ("cell_state", "log_probs", "finished", "lengths", "accumulated_attention_probs"))):
    pass

class BeamSearchDecoderOutput(
    collections.namedtuple("BeamSearchDecoderOutput", ("scores", "predicted_ids", "parent_ids"))):
    pass

class FinalBeamSearchDecoderOutput(
    collections.namedtuple("FinalBeamDecoderOutput", ["predicted_ids", "beam_search_decoder_output"])):
    pass

def gather_tree_from_array(t, parent_ids, sequence_length):
    max_time = parent_ids.shape[0].value or array_ops.shape(parent_ids)[0]
    batch_size = parent_ids.shape[1].value or array_ops.shape(parent_ids)[1]
    beam_width = parent_ids.shape[2].value or array_ops.shape(parent_ids)[2]

    # Generate beam ids that will be reordered by gather_tree.
    beam_ids = array_ops.expand_dims(
        array_ops.expand_dims(math_ops.range(beam_width), 0), 0)
    beam_ids = array_ops.tile(beam_ids, [max_time, batch_size, 1])

    max_sequence_lengths = math_ops.to_int32(
        math_ops.reduce_max(sequence_length, axis=1))
    sorted_beam_ids = beam_search_ops.gather_tree(
        step_ids=beam_ids,
        parent_ids=parent_ids,
        max_sequence_lengths=max_sequence_lengths,
        end_token=beam_width + 1)

    # For out of range steps, simply copy the same beam.
    in_bound_steps = array_ops.transpose(
        array_ops.sequence_mask(sequence_length, maxlen=max_time),
        perm=[2, 0, 1])
    sorted_beam_ids = array_ops.where(
        in_bound_steps, x=sorted_beam_ids, y=beam_ids)

    # Generate indices for gather_nd.
    time_ind = array_ops.tile(array_ops.reshape(
        math_ops.range(max_time), [-1, 1, 1]), [1, batch_size, beam_width])
    batch_ind = array_ops.tile(array_ops.reshape(
        math_ops.range(batch_size), [-1, 1, 1]), [1, max_time, beam_width])
    batch_ind = array_ops.transpose(batch_ind, perm=[1, 0, 2])
    indices = array_ops.stack([time_ind, batch_ind, sorted_beam_ids], -1)

    # Gather from a tensor with collapsed additional dimensions.
    gather_from = t
    final_shape = array_ops.shape(gather_from)
    gather_from = array_ops.reshape(
        gather_from, [max_time, batch_size, beam_width, -1])
    ordered = array_ops.gather_nd(gather_from, indices)
    ordered = array_ops.reshape(ordered, final_shape)

    return ordered

def _check_maybe(t):
    if isinstance(t, tensor_array_ops.TensorArray):
        raise TypeError(
            "TensorArray state is not supported by BeamSearchDecoder: %s" % t.name)
    if t.shape.ndims is None:
        raise ValueError(
            "Expected tensor (%s) to have known rank, but ndims == None." % t)

def _check_static_batch_beam_maybe(shape, batch_size, beam_width):
    reshaped_shape = tensor_shape.TensorShape([batch_size, beam_width, None])
    if (batch_size is not None and shape[0].value is not None
            and (shape[0] != batch_size * beam_width
                 or (shape.ndims >= 2 and shape[1].value is not None
                     and (shape[0] != batch_size or shape[1] != beam_width)))):
        return False
    return True

def _check_batch_beam(t, batch_size, beam_width):
    error_message = ("TensorArray reordering expects elements to be "
                     "reshapable to [batch_size, beam_size, -1] which is "
                     "incompatible with the dynamic shape of %s elements. "
                     "Consider setting reorder_tensor_arrays to False to disable "
                     "TensorArray reordering during the beam search."
                     % (t.name))
    rank = t.shape.ndims
    shape = array_ops.shape(t)
    if rank == 2:
        condition = math_ops.equal(shape[1], batch_size * beam_width)
    else:
        condition = math_ops.logical_or(
            math_ops.equal(shape[1], batch_size * beam_width),
            math_ops.logical_and(
                math_ops.equal(shape[1], batch_size),
                math_ops.equal(shape[2], beam_width)))
    return control_flow_ops.Assert(condition, [error_message])

class BeamSearchDecoder(decoder.Decoder):
    """BeamSearch sampling decoder."""
    def __init__(self,
                 cell,
                 embedding,
                 start_tokens,
                 end_token,
                 initial_state,
                 beam_width,
                 output_layer=None,
                 output_layer_2=None,
                 length_penalty_weight=0.0,
                 coverage_penalty_weight=0.0,
                 reorder_tensor_arrays=True,
                 attention_output_layer=None,
                 overlap_matrix=None):
        rnn_cell_impl.assert_like_rnncell("cell", cell)
        if (output_layer is not None
                and not isinstance(output_layer, layers_base.Layer)):
            raise TypeError(
                "output_layer must be a Layer, received: %s" % type(output_layer))
        self._cell = cell
        self._output_layer = output_layer
        self._output_layer_2 = output_layer_2
        self._attention_layer = attention_output_layer
        self._overlap_matrix = overlap_matrix
        self._reorder_tensor_arrays = reorder_tensor_arrays

        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self._start_tokens = ops.convert_to_tensor(
            start_tokens, dtype=dtypes.int32, name="start_tokens")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._end_token = ops.convert_to_tensor(
            end_token, dtype=dtypes.int32, name="end_token")
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")

        self._batch_size = array_ops.size(start_tokens)
        self._beam_width = beam_width
        self._length_penalty_weight = length_penalty_weight
        self._coverage_penalty_weight = coverage_penalty_weight
        self._initial_cell_state = nest.map_structure(self._maybe_split_batch_beams, initial_state, self._cell.state_size)
        self._start_tokens = array_ops.tile(array_ops.expand_dims(self._start_tokens, 1), [1, self._beam_width])
        self._start_inputs = self._embedding_fn(self._start_tokens)
        self._finished = array_ops.one_hot(array_ops.zeros([self._batch_size], dtype=dtypes.int32), depth=self._beam_width, on_value=False, off_value=True,
                                           dtype=dtypes.bool)

    @property
    def batch_size(self):
        return self._batch_size

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size
        else:
            output_shape_with_unknown_batch = nest.map_structure(lambda s: tensor_shape.TensorShape([None]).concatenate(s), size)
            layer_output_shape = self._output_layer.compute_output_shape(output_shape_with_unknown_batch)
            return nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def tracks_own_finished(self):
        return True

    @property
    def output_size(self):
        # Return the cell output and the id
        return BeamSearchDecoderOutput(
            scores=tensor_shape.TensorShape([self._beam_width]),
            predicted_ids=tensor_shape.TensorShape([self._beam_width]),
            parent_ids=tensor_shape.TensorShape([self._beam_width]))

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and int32 (the id)
        dtype = nest.flatten(self._initial_cell_state)[0].dtype
        return BeamSearchDecoderOutput(
            scores=nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            predicted_ids=dtypes.int32,
            parent_ids=dtypes.int32)

    def initialize(self, name=None):
        finished, start_inputs = self._finished, self._start_inputs
        dtype = nest.flatten(self._initial_cell_state)[0].dtype
        log_probs = array_ops.one_hot(  # shape(batch_sz, beam_sz)
            array_ops.zeros([self._batch_size], dtype=dtypes.int32),
            depth=self._beam_width,
            on_value=ops.convert_to_tensor(0.0, dtype=dtype),
            off_value=ops.convert_to_tensor(-np.Inf, dtype=dtype),
            dtype=dtype)
        init_attention_probs = get_attention_probs(self._initial_cell_state, self._coverage_penalty_weight)
        if init_attention_probs is None:
            init_attention_probs = ()
        initial_state = BeamSearchDecoderState(
            cell_state=self._initial_cell_state,
            log_probs=log_probs,
            finished=finished,
            lengths=array_ops.zeros(
                [self._batch_size, self._beam_width], dtype=dtypes.int64),
            accumulated_attention_probs=init_attention_probs)

        return (finished, start_inputs, initial_state)

    def finalize(self, outputs, final_state, sequence_lengths):
        del sequence_lengths
        max_sequence_lengths = math_ops.to_int32(math_ops.reduce_max(final_state.lengths, axis=1))
        predicted_ids = beam_search_ops.gather_tree(outputs.predicted_ids, outputs.parent_ids, max_sequence_lengths=max_sequence_lengths,
                                                    end_token=self._end_token)
        if self._reorder_tensor_arrays:
            final_state = final_state._replace(cell_state=nest.map_structure(lambda t: self._maybe_sort_array_beams(t, outputs.parent_ids, final_state.lengths),
                                                                             final_state.cell_state))
        outputs = FinalBeamSearchDecoderOutput(
            beam_search_decoder_output=outputs, predicted_ids=predicted_ids)
        return outputs, final_state

    def _merge_batch_beams(self, t, s=None):
        if isinstance(s, ops.Tensor):
            s = tensor_shape.as_shape(tensor_util.constant_value(s))
        else:
            s = tensor_shape.TensorShape(s)
        t_shape = array_ops.shape(t)
        static_batch_size = tensor_util.constant_value(self._batch_size)
        batch_size_beam_width = (
            None if static_batch_size is None
            else static_batch_size * self._beam_width)
        reshaped_t = array_ops.reshape(
            t, array_ops.concat(
                ([self._batch_size * self._beam_width], t_shape[2:]), 0))
        reshaped_t.set_shape(
            (tensor_shape.TensorShape([batch_size_beam_width]).concatenate(s)))
        return reshaped_t

    def _split_batch_beams(self, t, s=None):
        if isinstance(s, ops.Tensor):
            s = tensor_shape.TensorShape(tensor_util.constant_value(s))
        else:
            s = tensor_shape.TensorShape(s)
        t_shape = array_ops.shape(t)
        reshaped_t = array_ops.reshape(
            t, array_ops.concat(
                ([self._batch_size, self._beam_width], t_shape[1:]), 0))
        static_batch_size = tensor_util.constant_value(self._batch_size)
        expected_reshaped_shape = tensor_shape.TensorShape(
            [static_batch_size, self._beam_width]).concatenate(s)
        if not reshaped_t.shape.is_compatible_with(expected_reshaped_shape):
            raise ValueError("Unexpected behavior when reshaping between beam width "
                             "and batch size.  The reshaped tensor has shape: %s.  "
                             "We expected it to have shape "
                             "(batch_size, beam_width, depth) == %s.  Perhaps you "
                             "forgot to create a zero_state with "
                             "batch_size=encoder_batch_size * beam_width?"
                             % (reshaped_t.shape, expected_reshaped_shape))
        reshaped_t.set_shape(expected_reshaped_shape)
        return reshaped_t

    def _maybe_split_batch_beams(self, t, s):
        if isinstance(t, tensor_array_ops.TensorArray):
            return t
        _check_maybe(t)
        if t.shape.ndims >= 1:
            return self._split_batch_beams(t, s)
        else:
            return t

    def _maybe_merge_batch_beams(self, t, s):
        if isinstance(t, tensor_array_ops.TensorArray):
            return t
        _check_maybe(t)
        if t.shape.ndims >= 2:
            return self._merge_batch_beams(t, s)
        else:
            return t

    def _maybe_sort_array_beams(self, t, parent_ids, sequence_length):
        if not isinstance(t, tensor_array_ops.TensorArray):
            return t
        # pylint: disable=protected-access
        if (not t._infer_shape or not t._element_shape
                or t._element_shape[0].ndims is None
                or t._element_shape[0].ndims < 1):
            shape = (
                t._element_shape[0] if t._infer_shape and t._element_shape
                else tensor_shape.TensorShape(None))
            return t
        shape = t._element_shape[0]
        # pylint: enable=protected-access
        if not _check_static_batch_beam_maybe(
                shape, tensor_util.constant_value(self._batch_size), self._beam_width):
            return t
        t = t.stack()
        with ops.control_dependencies(
                [_check_batch_beam(t, self._batch_size, self._beam_width)]):
            return gather_tree_from_array(t, parent_ids, sequence_length)

    def step(self, time, inputs, state, name=None):
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token
        length_penalty_weight = self._length_penalty_weight
        coverage_penalty_weight = self._coverage_penalty_weight

        with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
            cell_state = state.cell_state
            inputs = nest.map_structure(
                lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
            cell_state = nest.map_structure(
                self._maybe_merge_batch_beams,
                cell_state, self._cell.state_size)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)

            cell_outputs = nest.map_structure(
                lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
            next_cell_state = nest.map_structure(
                self._maybe_split_batch_beams,
                next_cell_state, self._cell.state_size)

            if self._output_layer_2 is not None:
                hidden_size = 2000
                retrieval_attention = next_cell_state.attention[:, :, hidden_size:]
                weighted_sum = self._attention_layer(array_ops.concat([cell_outputs, retrieval_attention], axis=-1))
                weighted_sum = self._output_layer_2(weighted_sum)
                extra_p = tf.einsum('aij,jk->aik', weighted_sum, self._overlap_matrix)

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
                cell_outputs += extra_p

            beam_search_output, beam_search_state = _beam_search_step(
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state,
                batch_size=batch_size,
                beam_width=beam_width,
                end_token=end_token,
                length_penalty_weight=length_penalty_weight,
                coverage_penalty_weight=coverage_penalty_weight)

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids
            next_inputs = control_flow_ops.cond(
                math_ops.reduce_all(finished), lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))

        return (beam_search_output, beam_search_state, next_inputs, finished)

def _beam_search_step(time, logits, next_cell_state, beam_state, batch_size,
                      beam_width, end_token, length_penalty_weight,
                      coverage_penalty_weight):
    static_batch_size = tensor_util.constant_value(batch_size)

    # Calculate the current lengths of the predictions
    prediction_lengths = beam_state.lengths
    previously_finished = beam_state.finished
    not_finished = math_ops.logical_not(previously_finished)

    vocab_size = logits.shape[-1].value or array_ops.shape(logits)[-1]

    # Calculate the total log probs for the new hypotheses
    # Final Shape: [batch_size, beam_width, vocab_size]
    step_log_probs = nn_ops.log_softmax(logits)
    array_mask = [1.0] * vocab_size
    array_mask[0] = -float('inf')
    sample_mask_neg = tf.constant(np.asarray(array_mask), tf.float32)
    array_mask[0] = float('inf')
    sample_mask_pos = tf.constant(np.asarray(array_mask), tf.float32)

    def check(x):
        return tf.where(tf.greater(x[0], 0), x * sample_mask_neg, x * sample_mask_pos)

    def loop(x):
        return tf.map_fn(check, x)

    step_log_probs = tf.map_fn(loop, step_log_probs)

    step_log_probs = _mask_probs(step_log_probs, end_token, previously_finished)
    total_probs = array_ops.expand_dims(beam_state.log_probs, 2) + step_log_probs

    # Calculate the continuation lengths by adding to all continuing beams.
    vocab_size = logits.shape[-1].value or array_ops.shape(logits)[-1]
    lengths_to_add = array_ops.one_hot(
        indices=array_ops.fill([batch_size, beam_width], end_token),
        depth=vocab_size,
        on_value=np.int64(0),
        off_value=np.int64(1),
        dtype=dtypes.int64)
    add_mask = math_ops.to_int64(not_finished)
    lengths_to_add *= array_ops.expand_dims(add_mask, 2)
    new_prediction_lengths = (lengths_to_add + array_ops.expand_dims(prediction_lengths, 2))

    accumulated_attention_probs = None
    attention_probs = get_attention_probs(
        next_cell_state, coverage_penalty_weight)
    if attention_probs is not None:
        attention_probs *= array_ops.expand_dims(math_ops.to_float(not_finished), 2)
        accumulated_attention_probs = (
                beam_state.accumulated_attention_probs + attention_probs)

    # Calculate the scores for each beam
    scores = _get_scores(
        log_probs=total_probs,
        sequence_lengths=new_prediction_lengths,
        length_penalty_weight=length_penalty_weight,
        coverage_penalty_weight=coverage_penalty_weight,
        finished=previously_finished,
        accumulated_attention_probs=accumulated_attention_probs)

    time = ops.convert_to_tensor(time, name="time")
    # During the first time step we only consider the initial beam
    scores_flat = array_ops.reshape(scores, [batch_size, -1])

    # Pick the next beams according to the specified successors function
    next_beam_size = ops.convert_to_tensor(beam_width, dtype=dtypes.int32, name="beam_width")
    next_beam_scores, word_indices = nn_ops.top_k(scores_flat, k=next_beam_size)
    next_beam_scores.set_shape([static_batch_size, beam_width])
    word_indices.set_shape([static_batch_size, beam_width])

    # Pick out the probs, beam_ids, and states according to the chosen predictions
    next_beam_probs = _tensor_gather_helper(
        gather_indices=word_indices,
        gather_from=total_probs,
        batch_size=batch_size,
        range_size=beam_width * vocab_size,
        gather_shape=[-1],
        name="next_beam_probs")
    raw_next_word_ids = math_ops.mod(word_indices, vocab_size, name="next_beam_word_ids")
    next_word_ids = math_ops.to_int32(raw_next_word_ids)
    next_beam_ids = math_ops.to_int32(word_indices / vocab_size, name="next_beam_parent_ids")

    # Append new ids to current predictions
    previously_finished = _tensor_gather_helper(
        gather_indices=next_beam_ids,
        gather_from=previously_finished,
        batch_size=batch_size,
        range_size=beam_width,
        gather_shape=[-1])
    next_finished = math_ops.logical_or(
        previously_finished,
        math_ops.equal(next_word_ids, end_token),
        name="next_beam_finished")

    # Calculate the length of the next predictions.
    # 1. Finished beams remain unchanged
    # 2. Beams that are now finished (EOS predicted) remain unchanged
    # 3. Beams that are not yet finished have their length increased by 1
    lengths_to_add = math_ops.to_int64(math_ops.logical_not(previously_finished))
    next_prediction_len = _tensor_gather_helper(
        gather_indices=next_beam_ids,
        gather_from=beam_state.lengths,
        batch_size=batch_size,
        range_size=beam_width,
        gather_shape=[-1])
    next_prediction_len += lengths_to_add
    next_accumulated_attention_probs = ()
    if accumulated_attention_probs is not None:
        next_accumulated_attention_probs = _tensor_gather_helper(
            gather_indices=next_beam_ids,
            gather_from=accumulated_attention_probs,
            batch_size=batch_size,
            range_size=beam_width,
            gather_shape=[batch_size * beam_width, -1],
            name="next_accumulated_attention_probs")

    next_cell_state = nest.map_structure(
        lambda gather_from: _maybe_tensor_gather_helper(
            gather_indices=next_beam_ids,
            gather_from=gather_from,
            batch_size=batch_size,
            range_size=beam_width,
            gather_shape=[batch_size * beam_width, -1]),
        next_cell_state)

    next_state = BeamSearchDecoderState(
        cell_state=next_cell_state,
        log_probs=next_beam_probs,
        lengths=next_prediction_len,
        finished=next_finished,
        accumulated_attention_probs=next_accumulated_attention_probs)

    output = BeamSearchDecoderOutput(
        scores=next_beam_scores,
        predicted_ids=next_word_ids,
        parent_ids=next_beam_ids)

    return output, next_state

def get_attention_probs(next_cell_state, coverage_penalty_weight):
    if coverage_penalty_weight == 0.0:
        return None
    probs_per_attn_layer = []
    if isinstance(next_cell_state, attention_wrapper.AttentionWrapperState):
        probs_per_attn_layer = [attention_probs_from_attn_state(next_cell_state)]
    elif isinstance(next_cell_state, tuple):
        for state in next_cell_state:
            if isinstance(state, attention_wrapper.AttentionWrapperState):
                probs_per_attn_layer.append(attention_probs_from_attn_state(state))

    if not probs_per_attn_layer:
        raise ValueError(
            "coverage_penalty_weight must be 0.0 if no cell is attentional.")

    if len(probs_per_attn_layer) == 1:
        attention_probs = probs_per_attn_layer[0]
    else:
        attention_probs = [
            array_ops.expand_dims(prob, -1) for prob in probs_per_attn_layer]
        attention_probs = array_ops.concat(attention_probs, -1)
        attention_probs = math_ops.reduce_mean(attention_probs, -1)

    return attention_probs


def _get_scores(log_probs, sequence_lengths, length_penalty_weight, coverage_penalty_weight, finished, accumulated_attention_probs):
    length_penalty_ = _length_penalty(
        sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight)
    scores = log_probs / length_penalty_

    coverage_penalty_weight = ops.convert_to_tensor(
        coverage_penalty_weight, name="coverage_penalty_weight")
    if coverage_penalty_weight.shape.ndims != 0:
        raise ValueError("coverage_penalty_weight should be a scalar, "
                         "but saw shape: %s" % coverage_penalty_weight.shape)

    if tensor_util.constant_value(coverage_penalty_weight) == 0.0:
        return scores

    if accumulated_attention_probs is None:
        raise ValueError(
            "accumulated_attention_probs can be None only if coverage penalty is "
            "disabled.")

    # Add source sequence length mask before computing coverage penalty.
    accumulated_attention_probs = array_ops.where(
        math_ops.equal(accumulated_attention_probs, 0.0),
        array_ops.ones_like(accumulated_attention_probs),
        accumulated_attention_probs)

    # coverage penalty =
    #     sum over `max_time` {log(min(accumulated_attention_probs, 1.0))}
    coverage_penalty = math_ops.reduce_sum(
        math_ops.log(math_ops.minimum(accumulated_attention_probs, 1.0)), 2)
    # Apply coverage penalty to finished predictions.
    coverage_penalty *= math_ops.to_float(finished)
    weighted_coverage_penalty = coverage_penalty * coverage_penalty_weight
    # Reshape from [batch_size, beam_width] to [batch_size, beam_width, 1]
    weighted_coverage_penalty = array_ops.expand_dims(
        weighted_coverage_penalty, 2)
    return scores + weighted_coverage_penalty

def attention_probs_from_attn_state(attention_state):
    # Attention probabilities over time steps, with shape
    # `[batch_size, beam_width, max_time]`.
    attention_probs = attention_state.alignments
    if isinstance(attention_probs, tuple):
        attention_probs = [
            array_ops.expand_dims(prob, -1) for prob in attention_probs]
        attention_probs = array_ops.concat(attention_probs, -1)
        attention_probs = math_ops.reduce_mean(attention_probs, -1)
    return attention_probs

def _length_penalty(sequence_lengths, penalty_factor):
    penalty_factor = ops.convert_to_tensor(penalty_factor, name="penalty_factor")
    penalty_factor.set_shape(())  # penalty should be a scalar.
    static_penalty = tensor_util.constant_value(penalty_factor)
    if static_penalty is not None and static_penalty == 0:
        return 1.0
    return math_ops.div((5. + math_ops.to_float(sequence_lengths))
                        ** penalty_factor, (5. + 1.) ** penalty_factor)

def _mask_probs(probs, eos_token, finished):
    vocab_size = array_ops.shape(probs)[2]
    finished_row = array_ops.one_hot(
        eos_token,
        vocab_size,
        dtype=probs.dtype,
        on_value=ops.convert_to_tensor(0., dtype=probs.dtype),
        off_value=probs.dtype.min)
    finished_probs = array_ops.tile(
        array_ops.reshape(finished_row, [1, 1, -1]),
        array_ops.concat([array_ops.shape(finished), [1]], 0))
    finished_mask = array_ops.tile(
        array_ops.expand_dims(finished, 2), [1, 1, vocab_size])

    return array_ops.where(finished_mask, finished_probs, probs)


def _maybe_tensor_gather_helper(gather_indices, gather_from, batch_size,
                                range_size, gather_shape):
    if isinstance(gather_from, tensor_array_ops.TensorArray):
        return gather_from
    _check_maybe(gather_from)
    if gather_from.shape.ndims >= len(gather_shape):
        return _tensor_gather_helper(
            gather_indices=gather_indices,
            gather_from=gather_from,
            batch_size=batch_size,
            range_size=range_size,
            gather_shape=gather_shape)
    else:
        return gather_from

def _tensor_gather_helper(gather_indices,
                          gather_from,
                          batch_size,
                          range_size,
                          gather_shape,
                          name=None):
    with ops.name_scope(name, "tensor_gather_helper"):
        range_ = array_ops.expand_dims(math_ops.range(batch_size) * range_size, 1)
        gather_indices = array_ops.reshape(gather_indices + range_, [-1])
        output = array_ops.gather(
            array_ops.reshape(gather_from, gather_shape), gather_indices)
        final_shape = array_ops.shape(gather_from)[:1 + len(gather_shape)]
        static_batch_size = tensor_util.constant_value(batch_size)
        final_static_shape = (
            tensor_shape.TensorShape([static_batch_size]).concatenate(
                gather_from.shape[1:1 + len(gather_shape)]))
        output = array_ops.reshape(output, final_shape, name="output")
        output.set_shape(final_static_shape)
        return output
