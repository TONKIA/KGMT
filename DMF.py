import tensorflow as tf


class DMF(tf.keras.models.Model):
    def __init__(self, user_layer, item_layer, embedding):
        super(DMF, self).__init__()
        self.user_layer = user_layer
        self.item_layer = item_layer
        self.embedding = embedding

    def build(self, input_shape):
        self.user_embedding = tf.Variable(self.embedding, trainable=False, name="user_embedding")
        self.item_embedding = tf.Variable(self.embedding.T, trainable=False, name="item_embedding")

        self.user_mlp = [tf.keras.layers.Dense(dim, activation="relu", name="user_mlp_" + str(idx)) for (idx, dim) in
                         enumerate(self.user_layer)]
        self.item_mlp = [tf.keras.layers.Dense(dim, activation="relu", name="item_mlp_" + str(idx)) for (idx, dim) in
                         enumerate(self.item_layer)]
        super(DMF, self).build(input_shape)

    @tf.function
    def call(self, x):
        # (batch, 4)
        # (batch, 1)
        uid = tf.cast(x[:, 0], dtype=tf.int32)
        vid = tf.cast(x[:, 1], dtype=tf.int32)

        user = tf.nn.embedding_lookup(self.user_embedding, uid)
        item = tf.nn.embedding_lookup(self.item_embedding, vid)

        for layer in self.user_mlp:
            user = layer(user)

        for layer in self.item_mlp:
            item = layer(item)

        norm_user = tf.sqrt(tf.reduce_sum(tf.square(user), axis=1))
        norm_item = tf.sqrt(tf.reduce_sum(tf.square(item), axis=1))

        y = tf.reduce_sum(tf.multiply(user, item), axis=1) / (norm_user * norm_item)
        return y
