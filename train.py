#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import cPickle
import Config
import Model

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

#file = sys.argv[1]
file = 'kobe.txt'
data = open(file,'r').read()
data = data.decode('utf-8')
chars = list(set(data)) #char vocabulary

data_size, _vocab_size = len(data), len(chars)      #
print 'data has %d characters, %d unique.' % (data_size, _vocab_size)
char_to_idx = { ch : i for i,ch in enumerate(chars) }
idx_to_char = { i : ch for i,ch in enumerate(chars) }

config = Config.Config()
config.vocab_size = _vocab_size

#将2个字典序列化，保存为model.voc方便以后存取使用
cPickle.dump((char_to_idx, idx_to_char), open(config.model_path+'.voc','w'), protocol=cPickle.HIGHEST_PROTOCOL)

#将data中的每个字符存为对应的index
context_of_idx = [char_to_idx[ch] for ch in data]

def data_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]#data的shape是(batch_size, batch_len)，每一行是连贯的一段，一次可输入多个段

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]#y就是x的错一位，即下一个词
        yield (x, y)

def run_epoch(session, m, data, eval_op):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps   #// "来表示整数除法，返回不大于结果的一个最大的整数
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()

    for step, (x, y) in enumerate(data_iterator(data, m.batch_size,
                                                    m.num_steps)):
        cost, state,rs,  _ = session.run([m.cost, m.final_state,m.merged,eval_op],#x和y的shape都是(batch_size, num_steps)
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})


        costs += cost
        iters += m.num_steps
       # rs = session.run(m.loss_op,feed_dict = {m.input_data: x,m.targets: y})
       # print "loss is %.4f"% cost
      # if step and step % (epoch_size // 10) == 0:
       #    print"%.2f perplexity: %.3f cost-time: %.2f s" %
      #         step * 1.0 / epoch_size, np.exp(costs / iters),
     #           (time.time() - start_time)
    #       start_time = time.time()

   # train_writer.add_summar
#    rs2 = session.run([m.merged2],{m.sumcost:costs,m.iter:iters})
    #print(type(rs))
    #print(type(rs2))
    print"loss is %.4f " %(costs/iters)

    return np.exp(costs / iters),rs
def main(_):
    train_data = context_of_idx

    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        #with tf.name_scope("Train") as train_scope:
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model.Model(is_training=True, config=config)



        tf.global_variables_initializer().run()

       # train_writer = tf.summary.FileWriter('summary', session.graph)

        #tf.summary.merge_all()
        model_saver = tf.train.Saver(tf.global_variables())
        #tf.summary.FileWriter('summary/', session.graph)
        writer = tf.summary.FileWriter('summary/', session.graph)
        #writer2 = tf.summary.FileWriter('summary/', session.graph)
        for i in range(config.iteration):
            print"Training Epoch: %d ..." % (i+1)
            train_perplexity,rs = run_epoch(session, m, train_data, m.train_op)
           # train_writer.add_summary(train_perplexity,)
            if (i+1) % 5 ==0:
                writer.add_summary(rs,i+1)

            print"Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity)
         #   with tf.name_scope('perplexity'):
         #       tf.summary.scalar('perplexity',train_perplexity)
            if (i+1) % config.save_freq == 0:
                print 'model saving ...'
                model_saver.save(session, config.model_path+'-%d'%(i+1))
                print 'Done!'


if __name__ == "__main__":
    tf.app.run()
