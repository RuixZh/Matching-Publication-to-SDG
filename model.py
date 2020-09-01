import tensorflow as tf
import os
import sys
from time import sleep
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.keras.preprocessing import sequence, text
import numpy as np
import input_helpers as iph
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import math
import operator
import time
import xarray as xr
import itertools
fp = 'all_SDG.csv'
flp = 'SDG_Overview.csv'
t_length = 25
a_length = 250
k_length = 10
sequence_length = t_length+a_length+k_length
bs = 256
epoches = 6
num_checkpoints = 5
embedding_size = 100
fc_size = 128
hidden_units = 64
dropout_keep_prob = 0.5
learning_rate = 0.001


class Trans_Hier_Matching(object):
    def __init__(self, tficf_goal, tficf_target, tficf_indicator, goal, target, indicator, num_classes, vocab_size):
        with tf.name_scope('Input'):
            self.input_t = tf.placeholder(tf.int32, [None, t_length])  ## convolution
            self.input_a = tf.placeholder(tf.int32, [None, a_length])
            self.input_k = tf.placeholder(tf.int32, [None, k_length])
            self.input_y = tf.placeholder(tf.float32, [None, num_classes])
            self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            tficf_goal = tf.cast(tficf_goal, tf.float32)
            tficf_target = tf.cast(tficf_target, tf.float32)
            tficf_indicator = tf.cast(tficf_indicator, tf.float32)
            self.num_classes = num_classes

        with tf.name_scope('Embedding'):
            embedding = tf.Variable(
                tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            padding_emb = tf.constant(0.0, dtype=tf.float32, shape=[1, embedding_size])
            emb = tf.concat([padding_emb, embedding], 0)
            t_emb = tf.nn.embedding_lookup(emb, self.input_t)
            a_emb = tf.nn.embedding_lookup(emb, self.input_a)
            k_emb = tf.nn.embedding_lookup(emb, self.input_k)
            goal_emb = tf.nn.embedding_lookup(emb, goal)
            target_emb = tf.nn.embedding_lookup(emb, target)
            indicator_emb = tf.nn.embedding_lookup(emb, indicator)
        goal_label = tf.tensordot(tficf_goal, tf.squeeze(goal_emb), axes=1)
        target_label = tf.tensordot(tficf_target, tf.squeeze(target_emb), axes=1)
        indicator_label = tf.tensordot(tficf_indicator, tf.squeeze(indicator_emb), axes=1)
        self.goal_len = goal_label.get_shape().as_list()[0]
        self.target_len = target_label.get_shape().as_list()[1]
        self.indicator_len = indicator_label.get_shape().as_list()[2]

        rnn_t_out = self.bi_rnn(t_emb, scope='title')
        rnn_a_out = self.bi_rnn(a_emb, scope='abs')
        rnn_k_out = self.bi_rnn(k_emb, scope='kws')
        rnn_out = tf.concat([rnn_t_out, rnn_a_out, rnn_k_out], axis=1)

        self.HLAN(rnn_out, goal_label, target_label, indicator_label)

        self.logits = tf.nn.sigmoid(self.g_out)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_out, labels=self.input_y)
        self.losses = tf.reduce_mean(tf.reduce_sum(losses, axis=1))
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(losses, global_step=self.global_step)

    def bi_rnn(self, inputs, scope=None):
        with tf.variable_scope(scope or 'BiRNN'):
            fw_cells = tf.nn.rnn_cell.LSTMCell(hidden_units)
            bw_cells = tf.nn.rnn_cell.LSTMCell(hidden_units)

            fw_cells = rnn_cell.DropoutWrapper(fw_cells, output_keep_prob=1 - self.dropout_rate)
            bw_cells = rnn_cell.DropoutWrapper(bw_cells, output_keep_prob=1 - self.dropout_rate)

            rnn_outputs, _ = rnn.bidirectional_dynamic_rnn(
                cell_fw=fw_cells,
                cell_bw=bw_cells,
                inputs=inputs,
                dtype=tf.float32
            )
            H = tf.concat(rnn_outputs, axis=2)  # 2 * hidden_units
        return H

    def HLAN(self, inputs, goal, target, indicator):

        # l2_inputs = tf.nn.l2_normalize(inputs, axis=-1)
        with tf.variable_scope('indicator'):
            indicator = tf.reshape(indicator, [self.goal_len*self.target_len*self.indicator_len, embedding_size])
            label = tf.layers.dense(indicator, 2 * hidden_units, use_bias=False, activation=tf.nn.tanh)  # (G*T*I, 2h)
            # l2_label = tf.nn.l2_normalize(label, axis=-1)
            label = tf.transpose(label)  # (2h, G*T*I)
            self.i_alphas = tf.nn.softmax(tf.tensordot(inputs, label, axes=1))  # (B, W, G*T*I)
            # i_max_pooling = tf.reduce_max(i_alphas, -1)  # (B, W, G*T)
            i_com = tf.matmul(tf.transpose(self.i_alphas,[0, 2, 1]), inputs)  # (B, G*T*I, 2h)
            i_fc_in = tf.layers.dense(i_com, fc_size, activation=tf.nn.relu)  # (B, G*T*I, fc)
            i_fc_out = tf.layers.dense(i_fc_in, 1, activation=None)  # (B, G*T*I, 1)
            self.i_out = tf.reshape(tf.reduce_sum(i_fc_out, -1), [-1, self.goal_len, self.target_len, self.indicator_len])  # (B, G, T, I)
            # i_out = tf.reshape(i_out, [-1, self.goal_len, self.target_len, 2*hidden_units])  # (B, G, T, 2h)
            self.i_out = tf.nn.sigmoid(self.i_out)
            i_transfer = tf.reduce_max(self.i_out, -1)  # (B, G, T)

        with tf.variable_scope('target'):
            target = tf.multiply(tf.expand_dims(i_transfer, -1), target)  # (B, G, T, d)
            label = tf.reshape(target, [-1, self.goal_len*self.target_len, embedding_size])  # (B, G*T, d)
            label = tf.layers.dense(label, 2 * hidden_units, use_bias=False, activation=tf.nn.tanh)  # (B, G*T, 2h)
            label = tf.transpose(label, [0, 2, 1])  # (B, 2h, G*T)
            self.t_alphas = tf.nn.softmax(tf.matmul(inputs, label))  # (B, W, G*T)
            # t_alphas = tf.reshape(t_alphas, [-1, sequence_length, self.goal_len, self.target_len])
            # t_max_pooling = tf.reduce_max(t_alphas, -1)  # (B, W, G)
            t_com = tf.matmul(tf.transpose(self.t_alphas,[0, 2, 1]), inputs)  # (B, G*T, 2h)
            t_fc_in = tf.layers.dense(t_com, fc_size, activation=tf.nn.relu)  # (B, G*T, fc)
            t_fc_out = tf.layers.dense(t_fc_in, 1, activation=None)  # (B, G*T, 1)
            self.t_out = tf.reshape(tf.reduce_sum(t_fc_out, -1), [-1, self.goal_len, self.target_len])  # (B, G, T)
            # i_out = tf.reshape(i_out, [-1, self.goal_len, self.target_len, 2*hidden_units])  # (B, G, T, 2h)
            self.t_out = tf.nn.sigmoid(self.t_out)
            t_transfer = tf.reduce_max(self.t_out, -1)  # (B, G)

        with tf.variable_scope('goal'):
            goal = tf.multiply(tf.expand_dims(t_transfer, -1), goal)  # (B, G, d)
            label = tf.layers.dense(goal, 2 * hidden_units, use_bias=False, activation=tf.nn.tanh)  # (B, G, 2h)
            # l2_label = tf.nn.l2_normalize(label, axis=-1)
            label = tf.transpose(label, [0, 2, 1])   # (B, 2h, G)
            self.g_alphas = tf.nn.softmax(tf.matmul(inputs, label))  # (B, W, G)
            g_com = tf.matmul(tf.transpose(self.g_alphas, [0, 2, 1]), inputs)  # (B, G, 2h)
            g_fc_in = tf.layers.dense(g_com, fc_size, activation=tf.nn.relu)    # (B, G, fc)
            g_fc_out = tf.layers.dense(g_fc_in, 1, activation=None)  # (B, G, 1)
            self.g_out = tf.reduce_sum(g_fc_out, -1)  # (B, G)

val_result = {}
for no in range(3,5):
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    tficf_indicator, tficf_target, tficf_goal, indicator, target, goal = iph.load_label()
    train_t, train_a, train_k, train_y, label = iph.load_data('data/train_'+str(no)+'.csv')
    test_t, test_a, test_k, test_y, _ = iph.load_data('data/test_'+str(no)+'.csv')

    tokenizer = text.Tokenizer(num_words=400000)
    tokenizer.fit_on_texts(goal + target + indicator + list(train_t) + list(train_a) + list(train_k))
    vocab_size = len(tokenizer.word_index)
    print(vocab_size)
    ########## title ###########
    train_t = tokenizer.texts_to_sequences(train_t)
    train_t = sequence.pad_sequences(train_t, maxlen=t_length, padding='post', truncating='post')
    test_t = tokenizer.texts_to_sequences(test_t)
    test_t = sequence.pad_sequences(test_t, maxlen=t_length, padding='post', truncating='post')

    ######### abstract ##########
    train_a = tokenizer.texts_to_sequences(train_a)
    train_a = sequence.pad_sequences(train_a, maxlen=a_length, padding='post', truncating='post')
    test_a = tokenizer.texts_to_sequences(test_a)
    test_a = sequence.pad_sequences(test_a, maxlen=a_length, padding='post', truncating='post')

    ######### keyword ##########
    train_k = tokenizer.texts_to_sequences(train_k)
    train_k = sequence.pad_sequences(train_k, maxlen=k_length, padding='post', truncating='post')
    test_k = tokenizer.texts_to_sequences(test_k)
    test_k = sequence.pad_sequences(test_k, maxlen=k_length, padding='post', truncating='post')

    train_y, test_y = np.array(train_y), np.array(test_y)

    ######### label word ##########
    goal = tokenizer.texts_to_sequences(goal)
    target = tokenizer.texts_to_sequences(target)
    indicator = tokenizer.texts_to_sequences(indicator)
    num_classes = train_y.shape[-1]

    batches = [(train_t[i: i+bs], train_a[i: i+bs], train_k[i: i+bs], train_y[i: i+bs])
               for i in range(0, len(train_t), bs)]
    test_batches = [(test_t[i: i+bs], test_a[i: i+bs], test_k[i: i+bs], test_y[i: i+bs])
                    for i in range(0, len(train_t), bs)]
    with tf.Graph().as_default(), tf.device('/gpu:0'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)

        session_conf.gpu_options.allow_growth = True

    sess = tf.Session(config=session_conf)

    print("started session")
    with sess.as_default():
        Model = Trans_Hier_Matching(
            tficf_goal=tficf_goal,
            tficf_target=tficf_target,
            tficf_indicator=tficf_indicator,
            goal=goal,
            target=target,
            indicator=indicator,
            num_classes=train_y.shape[-1],
            vocab_size=vocab_size
        )
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        train_batch_per_epoch = int((len(train_t) - 1) / bs) + 1
        max_f1 = -0.0
        for epoch in range(epoches):
            tr_sc = []
            for i, (t, a, k, y) in enumerate(batches):
                percent = 100 * (i + 1) / train_batch_per_epoch
                sys.stdout.write('\r')
                sys.stdout.write("Epoch {0:d}: Batch {1:.2f}%".format(epoch + 1, percent))
                sys.stdout.flush()
                sleep(0.1)
                feed_dict = {Model.input_t: t,
                             Model.input_a: a,
                             Model.input_k: k,
                             Model.input_y: y,
                             Model.dropout_rate: 1 - dropout_keep_prob,
                             Model.is_training: True}
                _, l, train_sc = sess.run([Model.optimizer, Model.losses, Model.logits], feed_dict)
                for j in train_sc:
                    tr_sc.append(j)

            g_scores = []


            ts_loss = 0.0
            print('\n==========Validation start===========')
            length = len(test_batches)
            for i, (t, a, k, y) in enumerate(test_batches):
                percent = 100 * (i+1)/length
                sys.stdout.write('\r')
                sys.stdout.write("Batch {0:.2f}%".format(percent))
                sys.stdout.flush()
                sleep(0.1)
                feed_dict = {Model.input_t: t,
                             Model.input_a: a,
                             Model.input_k: k,
                             Model.input_y: y,
                             Model.dropout_rate: 0.0,
                             Model.is_training: False}
                l, g_sc, i_sc, t_sc, i_a, t_a, g_a = sess.run([Model.losses, Model.logits, Model.i_out, Model.t_out, Model.i_alphas, Model.t_alphas, Model.g_alphas], feed_dict)
                ts_loss += l
                for j in range(len(g_sc)):
                    g_scores.append(g_sc[j])
                if i == 0:
                    i_p = i_sc
                    t_p = t_sc
                    i_ = i_a
                    t_ = t_a
                    g_ = g_a

            print('\n==============Metrics============')
            ts_loss = ts_loss/len(test_batches)
            print('Test Loss: {0:.2f}'.format(ts_loss))
            em = iph.grid_search(test_y, g_scores)
            current_step = tf.train.global_step(sess, Model.global_step)
            if em[2] > max_f1:
                max_f1 = em[2]
                g_pred = g_scores
                i_pred = i_p
                t_pred = t_p
                i_alpha = i_
                t_alpha = t_
                g_alpha = g_
                max_em = em
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {0}\n".format(path))
            print('==============Epoch {} End============'.format(epoch+1))
        print('Best metrics:\nprecision: {0[0]}, recall: {0[1]}, f1: {0[2]}, roc_uc: {0[3]}, pr_auc: {0[4]}'.format(max_em))
        val_result[no] = max_em
        np.save("g_pred.npy", np.array(g_pred))
        np.save("i_pred.npy", np.array(i_pred))
        np.save("t_pred.npy", np.array(t_pred))
        np.save("i_alpha.npy", np.array(i_alpha))
        np.save("t_alpha.npy", np.array(t_alpha))
        np.save("g_alpha.npy", np.array(g_alpha))


        # p = xr.DataArray(data=np.array(prediction), coords=[label], dims=['label'])
        #p.to_series().unstack().to_csv('pred/prediction_'+str(no)+'.csv', index=True, header=True)
em = np.array([np.array(x) for x in val_result.values()]).mean(axis=0)
print('Average Evaluation:\nprecision: {0[0]}, recall: {0[1]}, f1: {0[2]}, roc_uc: {0[3]}, pr_auc: {0[4]}'.format(em))
