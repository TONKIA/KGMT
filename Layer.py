"""
Knowledge Graph Multi-Target CDR Model, KGMT
@author: TONKIA
@github: https://github.com/TONKIA
@email: tongxk96@outlook.com
"""

import tensorflow as tf


class MultipleMerge(tf.keras.layers.Layer):
    def __init__(self, units=8, regularizer=None, **kwargs):
        super(MultipleMerge, self).__init__(**kwargs)
        self.units = units
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        self.last_dim = input_shape[-1][-1]
        self.merge_block = []
        for idx in range(self.units):
            self.merge_block.append({
                "wq": self.add_weight(name="wq" + str(idx), shape=(2 * self.last_dim, self.last_dim),
                                      initializer="glorot_uniform",
                                      regularizer=self.regularizer,
                                      trainable=True),
                "wk": self.add_weight(name="wk" + str(idx), shape=(self.last_dim, self.last_dim),
                                      initializer="glorot_uniform",
                                      regularizer=self.regularizer,
                                      trainable=True),
                "wv": self.add_weight(name="wv" + str(idx), shape=(self.last_dim, self.last_dim),
                                      initializer="glorot_uniform",
                                      regularizer=self.regularizer,
                                      trainable=True),
            })
        self.wo = tf.keras.layers.Dense(self.last_dim, kernel_regularizer=self.regularizer, name="wo")
        super(MultipleMerge, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        # (batch, k)
        p1 = inputs[0]
        p2 = inputs[1]
        # (batch, 1, 2k)
        p1p2 = tf.expand_dims(tf.concat([p1, p2], axis=1), 1)
        # (batch, 2, k)
        p1p2_kv = tf.concat([tf.expand_dims(p1, 1), tf.expand_dims(p2, 1)], axis=1)
        wo_list = []
        for block in self.merge_block:
            # (batch, 1, k)
            q = tf.matmul(p1p2, block["wq"])
            # (batch, 2, k)
            k = tf.matmul(p1p2_kv, block["wk"])
            # (batch, 2, k)
            v = tf.matmul(p1p2_kv, block["wv"])
            # (batch, 1, 2)
            qk = tf.matmul(q, v, transpose_b=True)
            # (batch, 1, 2)
            qk_norm = tf.nn.softmax(qk / tf.math.sqrt(float(self.last_dim)))
            # (batch, k)
            qkv = tf.squeeze(tf.matmul(qk_norm, v))
            wo_list.append(qkv)
        qkv_concat = tf.reshape(tf.concat(wo_list, axis=1), [-1, self.units * self.last_dim])
        return self.wo(qkv_concat)
