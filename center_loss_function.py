from tensorflow.python.keras import backend as K
import tensorflow as tf

class CenterLossLayer(tf.keras.layers.Layer):
    def __init__(self, n_classes, n_features, **kwargs):
        super(CenterLossLayer, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.n_features = n_features
        self.centers = tf.Variable(
            tf.zeros([n_classes, n_features]),
            name="centers",
            trainable=False,
            # in a distributed strategy, we want updates to this variable to be summed.
            aggregation=tf.VariableAggregation.SUM)

    def call(self, x):
        # pass through layer
        return tf.identity(x)

    def get_config(self):
        config = super().get_config()
        config.update({"n_classes": self.n_classes, "n_features": self.n_features})
        return config


class CenterLoss(tf.keras.losses.Loss):

    def __init__(
            self,
            centers_layer,
            alpha=0.5,
            update_centers=True,
            name="center_loss",
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.centers_layer = centers_layer
        self.centers = self.centers_layer.centers
        self.alpha = alpha
        self.update_centers = update_centers

        # def call(self, sparse_labels, prelogits):
        #     sparse_labels = tf.reshape(sparse_labels, (-1,))
        #     centers_batch = tf.gather(self.centers, sparse_labels)
        #     # the reduction of batch dimension will be done by the parent class
        #     center_loss = tf.keras.losses.mean_squared_error(prelogits, centers_batch)

        #     # update centers
        #     if self.update_centers:
        #         diff = (1 - self.alpha) * (centers_batch - prelogits)
        #         updates = tf.scatter_nd(sparse_labels[:, None], diff, self.centers.shape)
        #         # using assign_sub will make sure updates are added during distributed
        #         # training
        #         self.centers.assign_sub(updates)
        #     return center_loss

    def call(self, sparse_labels, prelogits):
        sparse_labels = tf.reshape(sparse_labels, (-1,))
        centers_batch = tf.gather(self.centers, sparse_labels)
        # the reduction of batch dimension will be done by the parent class

        # 当前mini-batch的特征值与它们对应的中心值之间的差。features的维度也是[batch,16]，所以diff也是[batch,16].
        diff = centers_batch - prelogits

        unique_label, unique_idx, unique_count = tf.unique_with_counts(sparse_labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])#比如从[1,2,3,4]变成[[1],[2],[3],[4]]

        diff = diff / tf.cast((1 + appear_times), tf.float32)#diff维度依然是batch_size * 16
        diff = self.alpha * diff

        # update centers
        if self.update_centers:
            updates = tf.scatter_nd(sparse_labels[:, None], diff, self.centers.shape)
            # using assign_sub will make sure updates are added during distributed training
            self.centers.assign_sub(updates)

        loss = tf.reduce_mean(tf.abs(prelogits-centers_batch))
        return loss
