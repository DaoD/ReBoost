from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
from tensorflow.python.ops.array_ops import concat
from tensorflow.python.ops.math_ops import matmul
from tensorflow.python.ops.nn import softmax

__all__ = [
    "BasicDecoderOutput",
    "BasicDecoder",
]

class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
    pass

class BasicDecoderOutput_TW(
    collections.namedtuple("BasicDecoderOutput_TW", ("rnn_output", "sample_id", "weighted_sum"))):
    pass

class BasicDecoder(decoder.Decoder):
    """Basic sampling decoder."""
    def __init__(self, cell, helper, initial_state, output_layer=None, overlap_matrix=None, output_layer_2=None, attention_output_layer=None):
        rnn_cell_impl.assert_like_rnncell("cell", cell)
        if not isinstance(helper, helper_py.Helper):
            raise TypeError("helper must be a Helper, received: %s" % type(helper))
        if (output_layer is not None
                and not isinstance(output_layer, layers_base.Layer)):
            raise TypeError(
                "output_layer must be a Layer, received: %s" % type(output_layer))
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer
        self._overlap_matrix = overlap_matrix
        self._output_layer_2 = output_layer_2
        self._attention_layer = attention_output_layer

    @property
    def batch_size(self):
        return self._helper.batch_size

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size
        else:
            output_shape_with_unknown_batch = nest.map_structure(
                lambda s: tensor_shape.TensorShape([None]).concatenate(s),
                size)
            layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
                output_shape_with_unknown_batch)
            return nest.map_structure(lambda s: s[1:], layer_output_shape)

    def _rnn_output_size_2(self):
        size = self._cell.output_size
        if self._output_layer_2 is None:
            return size
        else:
            output_shape_with_unknown_batch = nest.map_structure(
                lambda s: tensor_shape.TensorShape([None]).concatenate(s),
                size)
            layer_output_shape = self._output_layer_2._compute_output_shape(  # pylint: disable=protected-access
                output_shape_with_unknown_batch)
            return nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def output_size(self):
        return BasicDecoderOutput_TW(
            rnn_output=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape,
            weighted_sum=self._rnn_output_size_2())

    @property
    def output_dtype(self):
        dtype = nest.flatten(self._initial_state)[0].dtype
        return BasicDecoderOutput_TW(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            self._helper.sample_ids_dtype,
            nest.map_structure(lambda _: dtype, self._rnn_output_size_2()), )

    def initialize(self, name=None):
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            hidden_size = 2000
            retrieval_attention = cell_state.attention[:, hidden_size:]
            weighted_sum = self._attention_layer(concat([cell_outputs, retrieval_attention], axis=-1))
            if self._output_layer_2 is not None:
                weighted_sum = self._output_layer_2(weighted_sum)
                extra_p = matmul(weighted_sum, self._overlap_matrix)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
                cell_outputs = softmax(cell_outputs) + softmax(extra_p)
            sample_ids = self._helper.sample(time=time, outputs=cell_outputs, state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)
        outputs = BasicDecoderOutput_TW(cell_outputs, sample_ids, weighted_sum)
        return (outputs, next_state, next_inputs, finished)
