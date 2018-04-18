#coding:utf-8
import tensorflow as tf

class Model(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self.lr = config.learning_rate

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #声明输入变量x, y

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=False)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/gpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size]) #size是wordembedding的维度
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)#返回一个tensor，shape是(batch_size, num_steps, size)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state) #inputs[:, time_step, :]的shape是(batch_size, size)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs,1), [-1, size])
        
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        self._final_state = state
        
        if not is_training:
            self._prob = tf.nn.softmax(logits)
            return
        with tf.name_scope('loss'):

            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(self._targets, [-1])],
                [tf.ones([batch_size * num_steps])],name = 'a')


            self.loss = tf.reduce_mean(loss)
            myloss = tf.summary.scalar("loss", self.loss)

        with tf.name_scope('cost'):
            self._cost = cost = tf.reduce_sum(loss) / batch_size
            mycost = tf.summary.scalar('cost',self._cost)
        #self.merged = tf.summary.merge_all()
        self.sumcost = tf.placeholder(tf.float32)
        self.iter = tf.placeholder(tf.float32)
      #  with tf.name_scope('perplexity'):
      #      self.perplexity = tf.exp(self.sumcost / self.iter)
      #      myperplexity = tf.summary.scalar('training perplexity',self.perplexity)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        self.merged = tf.summary.merge([myloss,mycost])
       # self.merged2 =tf.summary.merge_all()
    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def train_op(self):
        return self._train_op
