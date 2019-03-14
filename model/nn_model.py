import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Built off code similar to CS230 Project

class Model(object):
    """
    Abstracts a Tensorflow graph for a learning task.
    """
    def add_placeholders(self):
        """
        Adds placeholder variables to tensorflow computational graph.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """
        Creates the feed_dict for one step of training.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self, lstm_output=None):
        """
        Implements the core of the model that transforms a batch of input data into predictions.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_evaluation_op(self, pred):
        """
        Evaluates predicted labels against the actual labels.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """
        Adds Ops for the loss function to the computational graph.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """
        Sets up the training Ops.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """
        Perform one step of gradient descent on the provided batch of data.
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """
        Make predictions for the provided batch of data
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def calculate_metrics(self, labels, predictions):

        conf_matrix = tf.confusion_matrix(labels, predictions, num_classes=self.config.n_classes, dtype=tf.float32)
        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]
        TP = conf_matrix[1, 1]

        pos_precision = tf.divide(TP, tf.add(TP, FP))
        pos_precision = tf.where(tf.is_nan(pos_precision), 0., pos_precision)

        pos_recall = tf.divide(TP, tf.add(TP, FN))
        pos_recall = tf.where(tf.is_nan(pos_recall), 0., pos_recall)

        pos_f1 = tf.divide(tf.multiply(2.0, tf.multiply(pos_recall, pos_precision)), tf.add(pos_precision, pos_recall))
        pos_f1 = tf.where(tf.is_nan(pos_f1), 0., pos_f1)

        return pos_precision, pos_recall, pos_f1

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        
        self.evaluate = self.add_evaluation_op(self.pred)
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
