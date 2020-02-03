import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.helper import Helper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
import numpy as np

class MyGreedyEmbeddingHelper(Helper):
    """A helper for use during inference.
    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding, start_tokens, end_token, tgt_vocab_size=None):
        """Initializer.
        Args:
          embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`. The returned tensor
            will be passed to the decoder input.
          start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
          end_token: `int32` scalar, the token that marks end of decoding.
        Raises:
          ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
            scalar.
        """
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self._start_tokens = ops.convert_to_tensor(
            start_tokens, dtype=dtypes.int32, name="start_tokens")
        self._end_token = ops.convert_to_tensor(
            end_token, dtype=dtypes.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        self._start_inputs = self._embedding_fn(self._start_tokens)
        self._vocab_size = tgt_vocab_size

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self, name=None):
        finished = array_ops.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        array_mask = [1.0] * self._vocab_size
        array_mask[0] = -float('inf')
        sample_mask = tf.constant(np.asarray(array_mask), tf.float32)
        softmax_output = tf.nn.softmax(outputs)
        outputs = softmax_output * sample_mask
        sample_ids = math_ops.cast(math_ops.argmax(outputs, axis=-1), dtypes.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)