import collections
import functools
import math

import numpy as np

from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism

def _modified_score(processed_query, retrieval_score, keys):
    dtype = processed_query.dtype
    num_units = keys.shape[2].value or array_ops.shape(keys)[2]
    processed_query = array_ops.expand_dims(processed_query, 1)
    z = variable_scope.get_variable("attention_z", [1], dtype=dtype)
    v = variable_scope.get_variable("attention_v", [num_units], dtype=dtype)
    learned_score = math_ops.reduce_sum(v * math_ops.sigmoid(keys + processed_query), [2])
    return z * retrieval_score + (1 - z) * learned_score  # if gate

class ModifiedAttention(_BaseAttentionMechanism):
    def __init__(self, num_units, memory, memory_sequence_length, retrieval_score, probability_fn=None, score_mask_value=None, dtype=None,
                 name="ModifiedAttention"):
        if dtype is None:
            dtype = dtypes.float32
        wrapped_probability_fn = lambda score, _: score
        super(ModifiedAttention, self).__init__(
            query_layer=layers_core.Dense(
                num_units, name="query_layer", use_bias=False, dtype=dtype),
            memory_layer=layers_core.Dense(
                num_units, name="memory_layer", use_bias=False, dtype=dtype),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._name = name
        self._retrieval_score = retrieval_score

    def __call__(self, query, state):
        with variable_scope.variable_scope(None, "modified_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _modified_score(processed_query, self._retrieval_score, self._keys)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state
