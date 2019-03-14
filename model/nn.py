import os
import shutil
import sys
import time
import json
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from nn_model import Model

features_dir = '../data/'
summaries_dir = '../summaries'
summaries_train_dir = '../summaries/train'
summaries_dev_dir = '../summaries/dev'
summaries_loss_dir = '../summaries/loss'

# Built off code similar to CS230 project

class Config(object):
    """
    Holds model hyperparams and data information.
    """
    n_features = 654
    n_classes = 2
    n_layers = 5
    dropout = 0.0
    hidden_sizes = [654, 300, 150, 75, 30]
    batch_size = 512
    n_epochs = 5
    lr = 0.001
    fp_w = 1
    cutoff = 0.5


class FeedForwardModel(Model):
    """
    Implements a feedforward neural network
    """

    def add_placeholders(self):
        """
        Generates placeholder variables to represent the input tensors
        """
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.n_features])
        self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.n_classes])

    def create_feed_dict(self, inputs_batch, labels_batch):
        """
        Creates the feed_dict for the dependency parser.
        """
        feed_dict = {self.input_placeholder: inputs_batch, self.labels_placeholder: labels_batch}
        return feed_dict

    def add_prediction_op(self):
        """
        Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            pred = h + b2
        """
        x = self.input_placeholder
        # x = tf.contrib.layers.batch_norm(x, scale=True)
        cache = {}
        for n in range(1, self.config.n_layers):
            cache['W' + str(n)] = tf.get_variable('W' + str(n), shape=[self.config.hidden_sizes[n - 1], self.config.hidden_sizes[n]], initializer=tf.contrib.layers.xavier_initializer())
            cache['b' + str(n)] = tf.get_variable('b' + str(n), shape=[1, self.config.hidden_sizes[n]], initializer=tf.zeros_initializer())

        cache['h1'] = tf.nn.relu(tf.matmul(x, cache['W1']) + cache['b1'])
        for n in range(2, self.config.n_layers):
            cache['h' + str(n)] = tf.nn.relu(tf.matmul(cache['h' + str(n - 1)], cache['W' + str(n)]) + cache['b' + str(n)])
        U = tf.get_variable("U", shape=[self.config.hidden_sizes[-1], self.config.n_classes], initializer=tf.contrib.layers.xavier_initializer())
        b_last = tf.get_variable("b_last", shape=[1, self.config.n_classes], initializer=tf.zeros_initializer())
        pred = tf.matmul(cache['h' + str(self.config.n_layers - 1)], U) + b_last
        return pred



    def add_evaluation_op(self, pred):
        """
        Calculates accuracy and saliency from predicted labels
        """
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.labels_placeholder, pred, 1))

        dl_dx = tf.gradients(loss, self.input_placeholder)
        saliency = dl_dx[0]

        soft = tf.nn.softmax(pred)

        test_val = tf.cast(tf.greater(tf.reduce_max(soft, axis=1), self.config.cutoff), tf.int64)
        test_i = soft
        maxed = tf.greater(tf.reduce_max(soft, axis=1), self.config.cutoff)
        argmax = tf.cast(tf.argmax(soft, 1), tf.bool)
        y_hat = tf.cast(tf.logical_and(maxed, argmax), tf.int64)
        # y_hat = tf.argmax(soft, 1)
        y = tf.argmax(self.labels_placeholder, 1)
        accuracy = tf.to_float(tf.reduce_sum(tf.cast(tf.equal(y_hat, y), tf.int32))) / tf.to_float(tf.size(y))

        pos_precision, pos_recall, pos_f1 = self.calculate_metrics(y, y_hat)

        return (test_val, test_i, y_hat, y, accuracy, pos_precision, pos_recall, pos_f1, saliency)

    def add_loss_op(self, pred):
        """
        Adds Ops for the cross entropy loss.
        # """
        # soft = tf.nn.softmax(pred)
        # log_loss = -((0) + ((1 - self.labels_placeholder) * tf.log(1 - soft)))
        # loss = tf.convert_to_tensor(0.3, dtype=tf.float32)
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.labels_placeholder, pred, self.config.fp_w))

        # dl_dx = tf.gradients(loss, self.input_placeholder)
        # saliency = tf.linalg.norm(dl_dx[0], axis=1)

        return loss

    def add_training_op(self, loss):
        """
        Sets up the training Ops.
        """
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op = opt.minimize(loss)
        return train_op

    def evaluate_on_set(self, sess, inputs, labels):
        feed = self.create_feed_dict(inputs, labels_batch=labels)
        data, _ = sess.run([self.evaluate, self.pred], feed_dict=feed)
        return data

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        
        return loss

    def run_epoch(self, sess, train_examples, dev_set, train_writer, dev_writer, loss_writer, epoch):
        dev_inputs = dev_set[:,:-1]
        dev_labels = to_categorical(dev_set[:,-1], num_classes=2)
        train_inputs = train_examples[:,:-1]
        train_labels = to_categorical(train_examples[:,-1], num_classes=2)
        n_minibatches = 1 + len(train_examples) / self.config.batch_size
        prog = tf.keras.utils.Progbar(target=n_minibatches)
        
        
       
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y)
            
            loss_summary = tf.Summary()
            loss_summary.value.add(tag='loss', simple_value=loss)
            loss_writer.add_summary(loss_summary, epoch * n_minibatches + i)
            loss_writer.flush()

            prog.update(i + 1, [("train loss", loss)], force=i + 1 == n_minibatches)

        # print('\nEvaluating on train set',)
        # _, _, _, _, train_acc, _, _, _, _ = self.evaluate_on_set(sess, train_inputs, train_labels)
        # print('- train accuracy: {:.4f}'.format(train_acc))

        # train_summary = tf.Summary()
        # train_summary.value.add(tag='accuracy', simple_value=train_acc)
        # train_writer.add_summary(train_summary, epoch)

        

        print("Evaluating on dev set",)
        _, _, _, _, dev_acc, _, _, _, _ = self.evaluate_on_set(sess, dev_inputs, dev_labels)
        print("- dev accuracy: {:.4f}".format(dev_acc))

        dev_summary = tf.Summary()
        dev_summary.value.add(tag='accuracy', simple_value=dev_acc)
        dev_writer.add_summary(dev_summary, epoch)

        return dev_acc

    def fit(self, sess, saver, train_examples, dev_set, train_writer, dev_writer, loss_writer):
        best_dev_acc = 0.0
        for epoch in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_acc = self.run_epoch(sess, train_examples, dev_set, train_writer, dev_writer, loss_writer, epoch)
            # if dev_acc > best_dev_acc:
            #     best_dev_acc = dev_acc
            if saver:
                print("New best dev accuracy! Saving model in ../params/feedforward_weights/best.weights")
                saver.save(sess, '../params/feedforward_weights/best.weights')

            train_writer.flush()
            dev_writer.flush()
            print()

    def __init__(self, config):
        self.config = config
        self.build()

def minibatches(train_examples, batch_size):
    batches = []
    n_batches = int(len(train_examples) / batch_size)
    for i in range(n_batches):
        sample = train_examples[i * batch_size:(i + 1) * batch_size,:-1]
        label = to_categorical(train_examples[i * batch_size:(i + 1) * batch_size,-1], num_classes=2)
        batches.append((sample, label))
    sample = train_examples[n_batches * batch_size:,:-1]
    label = to_categorical(train_examples[n_batches * batch_size:,-1], num_classes=2)
    batches.append((sample, label))
    return batches

def main():
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    config = Config()

    train_examples = pd.read_csv('../model_data/train.csv', nrows=1000000, header=None).sample(n=1000000).values
    dev_set = pd.read_csv('../model_data/validation.csv', nrows=100000, header=None).sample(n=100000).values
    test_set = pd.read_csv('../model_data/test.csv', nrows=100000, header=None).sample(n=100000).values

    if not os.path.exists('../params/feedforward_weights/'):
        os.makedirs('../params/feedforward_weights/')

    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)

    if os.path.exists(summaries_train_dir):
        shutil.rmtree(summaries_train_dir)
    if os.path.exists(summaries_dev_dir):
        shutil.rmtree(summaries_dev_dir)
    if os.path.exists(summaries_loss_dir):
        shutil.rmtree(summaries_loss_dir)

    os.makedirs(summaries_train_dir)
    os.makedirs(summaries_dev_dir)
    os.makedirs(summaries_loss_dir)

    if not os.path.exists(summaries_train_dir):
        os.makedirs(summaries_train_dir)
    if not os.path.exists(summaries_dev_dir):
        os.makedirs(summaries_dev_dir)

    # fp_weights = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    lr = [0.1, 0.01, 0.001, 0.0001]
    cutoffs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9]
    all_acc = []
    all_pre = []
    all_rec = []
    all_f1 = []
    for c in cutoffs:
        config.cutoff = c
        with tf.Graph().as_default() as graph:
            print("Building model...",)
            start = time.time()
            model = FeedForwardModel(config)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            saver = tf.train.Saver()
            print("took {:.2f} seconds\n".format(time.time() - start))
        graph.finalize()

        with tf.Session(graph=graph) as session:
            train_writer = tf.summary.FileWriter(summaries_train_dir, session.graph)
            dev_writer = tf.summary.FileWriter(summaries_dev_dir, session.graph)
            loss_writer = tf.summary.FileWriter(summaries_loss_dir, session.graph)


            session.run(init_op)

            print(80 * "=")
            print("TRAINING")
            print(80 * "=")
            model.fit(session, saver, train_examples, dev_set, train_writer, dev_writer, loss_writer)

            print(80 * "=")
            print("TESTING")
            print(80 * "=")
            print("Restoring the best model weights found on the dev set")
            saver.restore(session, '../params/feedforward_weights/best.weights')
            print("Final evaluation on test set",)
            test_inputs = dev_set[:,:-1]
            test_labels = to_categorical(dev_set[:,-1], num_classes=2)
            test_val, test_i, y_hat, y, acc, pos_precision, pos_recall, pos_f1, saliency = model.evaluate_on_set(session, test_inputs, test_labels)
            print(test_i[:5])
            print(test_val[:5])
            print(y_hat[:5])
            print(y[:5])
            # for i in range(len(y_hat)):
            #     print(id_to_group[str(y_hat[-i])], id_to_group[str(y[-i])])
            print('- test accuracy: {:.4f}'.format(acc))
            print('- test precision: {:.4f}'.format(pos_precision))
            print('- test recall: {:.4f}'.format(pos_recall))
            print('- test f1: {:.4f}'.format(pos_f1))
            # if pos_precision > best_metric:
            #     best_lr = l
            #     best_metric = pos_precision
            #     best_acc = acc
            #     best_pre = pos_precision
            #     best_rec = pos_recall
            #     best_f1 = pos_f1
            all_acc.append(acc)
            all_pre.append(pos_precision)
            all_rec.append(pos_recall)
            all_f1.append(pos_f1)
    plt.title('Metrics Over Delayed Probability Cutoff')
    plt.ylabel('Metric')
    plt.xlabel('Delayed Probability Cutoff')
    # plt.xscale('log')
    plt.plot(cutoffs, all_acc, label='Accuracy')
    plt.plot(cutoffs, all_pre, label='Precision')
    plt.plot(cutoffs, all_rec, label='Recall')
    plt.plot(cutoffs, all_f1, label='F1 Score')
    plt.legend()
    plt.savefig('metrics_cutoff1.png')
    plt.show()




if __name__ == '__main__':
    main()
