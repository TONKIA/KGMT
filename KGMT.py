"""
Knowledge Graph Multi-Target CDR Model, KGMT
@author: TONKIA
@github: https://github.com/TONKIA
@email: tongxk96@outlook.com
"""

import tensorflow as tf
from Layer import MultipleMerge


class KGMT(tf.keras.models.Model):
    def __init__(self, dim, n_user, n_item, global_emb, unit=1, reg=0.001, mlp_dim=[256, 512, 1024, 512]):
        super(KGMT, self).__init__()
        self.dim = dim
        self.n_user = n_user
        self.n_item = n_item
        self.global_emb = global_emb
        self.reg = reg
        self.mlp_dim = mlp_dim
        self.unit = unit

    def build(self, input_shape):
        l2 = tf.keras.regularizers.l2(self.reg)
        self.item_emb = tf.keras.layers.Embedding(self.n_item, self.dim, embeddings_initializer='glorot_uniform',
                                                  embeddings_regularizer=l2, name="item_emb")
        self.user_emb = tf.keras.layers.Embedding(self.n_user, self.dim, embeddings_initializer='glorot_uniform',
                                                  embeddings_regularizer=l2, name="user_emb")
        self.global_emb = tf.Variable(self.global_emb, trainable=False, name="global_emb")
        self.mlp = [tf.keras.layers.Dense(dim, activation="relu", kernel_regularizer=l2, name="mlp_" + str(idx)) for
                    (idx, dim) in enumerate(self.mlp_dim)]
        self.predict_layer = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=l2, name="predict_layer")
        if self.unit > 0:
            self.multiple_merge_user = MultipleMerge(units=self.unit, regularizer=l2, name="multiple_merge_user")
            self.multiple_merge_item = MultipleMerge(units=self.unit, regularizer=l2, name="multiple_merge_item")
        super(KGMT, self).build(input_shape)

    @tf.function
    def call(self, x):
        # (batch, 4)
        # (batch, 1)
        uid = x[:, 0]
        vid = x[:, 1]
        guid = tf.cast(x[:, 2], dtype=tf.int32)
        gid = tf.cast(x[:, 3], dtype=tf.int32)
        # (batch, k)
        user = self.user_emb(uid)
        # (batch, k)
        item = self.item_emb(vid)
        # (batch, k)
        g_user = tf.nn.embedding_lookup(self.global_emb, guid)
        # (batch, k)
        g_item = tf.nn.embedding_lookup(self.global_emb, gid)
        # (batch, k)
        if self.unit > 0:
            merge_user = self.multiple_merge_user([user, g_user])
            merge_item = self.multiple_merge_item([item, g_item])
        else:
            merge_user = user
            merge_item = item
        # MLP
        y = tf.concat([merge_user, merge_item], axis=1)
        for layer in self.mlp:
            y = layer(y)
        y = self.predict_layer(y)
        return y
