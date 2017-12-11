# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:17:40 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:51:43 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:29:34 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:03:29 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:22:02 2017

@author: thy1995
"""

import tensorflow as tf
import numpy as np
import scipy.signal
import os
import random

import matplotlib.pyplot as plt
from datetime import datetime

from ros_env import RosEnv
print("debug")
tf.reset_default_graph()

def makeFolder(addr):
    if not os.path.exists(addr):
        os.makedirs(addr)

start_time = str(datetime.now())
clipped = True
new_shuffle = False
new_loss = True
UNROLL =  40
  

gamma = 0.9
num_discount = 20

scaled = False
noise = False

debug_mode = False

plt.rcParams["figure.figsize"] = [16, 9]

savefolder = "/home/thy/shareFolder/ros_ckpt/debug/"
#savefolder= '/media/thy/7A6E453B6E44F185/workspace/fwl_project/savefile'

text_file = open(savefolder + "readme.txt", "w")

text_file.write("New loss new shuffle clipped 16 nfg")

text_file.close()


cell_count = 1 
hidden_layer_size = 48

input_size = 16
target_size = 24
num_epoch = 3000
lr = 0.01
epoch_threshold = 500

        
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1][0]

def reward_batch(rewards, V, num_step):
    to_return = []
    for i in range(len(rewards)):
        temp = [rewards[i] for  i in range(i, min(i+num_step, len(rewards)))]
        discounted = discount(temp, gamma) - V[i] * (gamma  ** (num_step - i))
        to_return.append(discounted)
    return to_return 

 

with tf.device("/cpu:0"):
    _inputs = tf.placeholder(tf.float32,shape=[None, None, input_size],
                              name='inputs')
    y_ = tf.placeholder(tf.float32, [None, 1], name = "label")
     
    
    class LSTM_cell(object):
    
        """
        LSTM cell object which takes 3 arguments for initialization.
        input_size = Input Vector size
        hidden_layer_size = Hidden layer size
        target_size = Output vector size
    
        """
    
        def __init__(self, input_size, hidden_layer_size, target_size):
    
            # Initialization of given values
            self.input_size = input_size
            self.hidden_layer_size = hidden_layer_size
            self.target_size = target_size
    
            # Weights and Bias for input and hidden tensor
            self.Wi = tf.Variable(tf.zeros(
                [self.input_size, self.hidden_layer_size]), name = "Wi")
            self.Ui = tf.Variable(tf.zeros(
                [self.hidden_layer_size, self.hidden_layer_size]),name ="Ui")
            self.bi = tf.Variable(tf.zeros([self.hidden_layer_size]),name = "bi")
    
#            self.Wf = tf.Variable(tf.zeros(
#                [self.input_size, self.hidden_layer_size]), name = "Wf")
#            self.Uf = tf.Variable(tf.zeros(
#                [self.hidden_layer_size, self.hidden_layer_size]), name = "Uf")
#            self.bf = tf.Variable(tf.zeros([self.hidden_layer_size]), name = "bf")
    
            self.Wog = tf.Variable(tf.zeros(
                [self.input_size, self.hidden_layer_size]), name = "Wog")
            self.Uog = tf.Variable(tf.zeros(
                [self.hidden_layer_size, self.hidden_layer_size]), name = "Uog")
            self.bog = tf.Variable(tf.zeros([self.hidden_layer_size]), name ="bog")
    
            self.Wc = tf.Variable(tf.zeros(
                [self.input_size, self.hidden_layer_size]), name = "Wc")
            self.Uc = tf.Variable(tf.zeros(
                [self.hidden_layer_size, self.hidden_layer_size]), name = "Uc")
            self.bc = tf.Variable(tf.zeros([self.hidden_layer_size]), name = "bc")
    
            # Weights for output layers
            self.Wo = tf.Variable(tf.truncated_normal(
                [self.hidden_layer_size, self.target_size], mean=0, stddev=.01), name = "Wo")
            self.bo = tf.Variable(tf.truncated_normal(
                [self.target_size], mean=0, stddev=.01), name = "bo")
    
            # Placeholder for input vector with shape[batch, seq, embeddings]
            #self._inputs = tf.placeholder(tf.float32,
                                          #shape=[None, None, self.input_size],
                                          #name='inputs')
    
            # Processing inputs to work with scan function
            self.processed_input = process_batch_input_for_RNN(_inputs)
    
    
            self.initial_hidden = _inputs[:, 0, :]
            self.initial_hidden = tf.matmul(
                self.initial_hidden, tf.zeros([input_size, hidden_layer_size]))
    
            self.initial_hidden = tf.stack(
                [self.initial_hidden, self.initial_hidden])
        # Function for LSTM cell.
    
        def Lstm(self, previous_hidden_memory_tuple, x):
            """
            This function takes previous hidden state and memory
             tuple with input and
            outputs current hidden state.
            """
            previous_hidden_state, c_prev = tf.unstack(previous_hidden_memory_tuple)
    
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            ) #were sigmoid
    
             #Forget Gate
#            f = tf.sigmoid(
#                tf.matmul(x, self.Wf) +
#                tf.matmul(previous_hidden_state, self.Uf) + self.bf
#            ) # were sigmoid
#    
            f = 1
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            ) #were sigmoid
    
            # New Memory Cell, or block input
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )
    
            # Final Memory cell
            c = f * c_prev + i * c_
            if clipped:
                c = tf.clip_by_value(c, -1 ,1, name ="Clipped")
            
            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)
    
            return tf.stack([current_hidden_state, c])
    
        # Function for getting all hidden state.
        def get_states(self):
            """
            Iterates through time/ sequence to get all hidden state
            """
    
            # Getting all hidden state throuh time
            all_hidden_states = tf.scan(self.Lstm,
                                        self.processed_input,
                                        initializer=self.initial_hidden,
                                        name='states')
            all_hidden_states = all_hidden_states[:, 0, :, :]
    
            return all_hidden_states
    
        # Function to get output from a hidden layer
        def get_output(self, hidden_state):
            """
            This function takes hidden state and returns output
            """
            output = tf.tanh(tf.matmul(hidden_state, self.Wo) + self.bo)
    
            return output
    
        # Function for getting all output layers
        def get_outputs(self):
            """
            Iterating through hidden states to get outputs for all timestamp
            """
            all_hidden_states = self.get_states()
    
            all_outputs = tf.map_fn(self.get_output, all_hidden_states)
    
            return all_outputs
    
    
    # Function to convert batch input data to use scan ops of tensorflow.
    def process_batch_input_for_RNN(batch_input):
        """
        Process tensor of size [5,3,2] to [3,5,2]
        """
        batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
        X = tf.transpose(batch_input_) #in case we do batch processing
    
        return X
    
    
    class LSTM_layer(object):
        def __init__(self, cell_count ,input_size, hidden_layer_size, target_size):
            self.input_size = input_size
            self.hidden_layer_size = hidden_layer_size
            self.target_size = target_size
            self.cell_count = cell_count
            self.LSTM_list = []
            for i in range(self.cell_count):
                self.LSTM_list.append(LSTM_cell(self.input_size, self.hidden_layer_size, self.target_size))
        
        def output(self):
            output = []
            for  i in range(self.cell_count):
                output.append(self.LSTM_list[i].get_outputs()[-1])
            return tf.reshape(tf.transpose(output, perm = [1,0,2]), [1,-1, target_size * cell_count])
        def output_debug(self):
            output = []
            for  i in range(self.cell_count):
                output.append(self.LSTM_list[i].get_outputs()[-1])
            return output
    

    rnn = LSTM_layer(cell_count, input_size, hidden_layer_size, target_size)
    outputs = rnn.output()
    print("Intialize ROS environment")
    env = RosEnv("rocky3",10,np.random.uniform(2,5, size = 2), False)
    
    action_space = len(env.StateSpace())
    
    W2 = tf.Variable(tf.zeros([1,cell_count *target_size,action_space]), name = "Weight2")
    B2 = tf.Variable(tf.zeros([action_space]), name = "Bias2")
    y4 = tf.nn.softmax(tf.matmul(outputs, W2) + B2, name = "output2")
    
    one_hot_pl = tf.placeholder(tf.uint8, [None])
    one_hot = tf.one_hot(one_hot_pl, action_space)
    one_hot_a = tf.expand_dims(one_hot, dim = 0)
    debug = tf.multiply(one_hot_a, y4 ) 
    action_prob = tf.reduce_sum(tf.multiply(one_hot_a, y4 ),2 , keep_dims = True)    
    W3_v = tf.Variable(tf.random_uniform([ 1, cell_count *target_size,1], minval = 0, maxval = 1), name = "Weight3_v")
    B3_v = tf.Variable(tf.zeros([1]), name = "Bias3_v")
    y4_v = tf.matmul(outputs, W3_v) + B3_v    
    advantage =  y_ - y4_v

    loss  = tf.reduce_sum(tf.multiply(-tf.log(action_prob), advantage)) + tf.reduce_sum(0.5* tf.pow(advantage, 2)) + tf.reduce_sum(action_prob * tf.log(action_prob))
    
    
    rms = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
    optimizer = rms
    print("Initialize new session")
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    

    #INPUT
    
    
    best_loss = float("INFINITY")
    
    all_best_loss = []
    all_loss = []
    saver0 = tf.train.Saver(max_to_keep = None)
    saver0.export_meta_graph(savefolder + "lr" + str(lr) + 'rms' + 'unroll' + str(UNROLL) + '.meta')
    
    
    for epoch_i in range(num_epoch):
        print("Epoch ", epoch_i)
        env.goal = np.random.uniform(3,3, size = 2).tolist()
        env.AdjustThreshold()
        temp_debug = np.array([], ndmin = 3).reshape(1,0,input_size)
        observation = env.reset()
        
        counter = 0
        batch = np.array([]).reshape(0,UNROLL,input_size)
        loss_l = []
        discounted_loss = []
        action_list = []
        last_output_v = np.array([0] * action_space)
        temp_loss = 0
        temp_ = []
        v_tList = []
        #INPUT, LABEL = shuffle(input, label)
        while True:
            #Tricky batch sampling
            observation = list(observation) + [temp_loss] + last_output_v.tolist()
            observation = np.expand_dims(np.expand_dims(observation,0), 0)
            
            temp_debug = np.append(temp_debug, observation, axis  =  1 )
            
            #For batch
            while len(temp_debug[0]) < UNROLL:
                temp_debug = np.insert(temp_debug, 0, np.ones([1,1, input_size]), axis = 1)
            if counter > 0:
                temp_debug = np.delete(temp_debug, 0 , axis = 1)

            last_output_v, v_t  =sess.run([y4,y4_v], feed_dict = {_inputs : temp_debug})
            v_t = np.squeeze(v_t, axis = (0,1))[0] 
                        
            
            last_output = np.random.choice(np.squeeze(last_output_v), p = np.squeeze(last_output_v))
            a  =np.argmax(last_output == np.squeeze(last_output_v))
            last_output_v = (last_output == np.squeeze(last_output_v)).astype(int)
            
            temp_loss, observation, done, solve = env.action(a)
            loss_l.append(np.abs(temp_loss) )
            
            counter = counter + 1
            batch = np.append(batch, temp_debug, axis =0)
            discounted_loss.append(temp_loss)
            action_list.append(a)
            v_tList.append(v_t)
            if done:
                break
        
        m = np.mean(loss_l)
        s = np.std(loss_l)
        
        all_loss.append(m)
        
        print("####################################################")
        print("Mean:", m)
        print("Std:", s)
        
        p1 = plt.plot(range(len(loss_l)),  1 - np.array(loss_l))
        plt.legend(handles = p1)
        plt.show()
        
        if solve:
            best_loss  = m
            makeFolder(savefolder+"best\\")
            saver0.save(sess, savefolder + "best/model.ckpt", global_step = epoch_i, write_meta_graph= False)
            
            text_file = open(savefolder + "Epoch" + str(epoch_i)+".txt", "w")
            text_file.write(str(best_loss))
            text_file.close()
            all_best_loss.append(best_loss)
        discounted_loss = reward_batch(discounted_loss, v_tList, counter)
        discounted_loss = np.reshape(discounted_loss, [len(discounted_loss),1])
        if debug_mode:
            print("testing crash")
            print("PASS")
            a,ac, DEBUG   = sess.run([y4_v,action_prob, tf.multiply(-tf.log(action_prob), advantage)], feed_dict = {_inputs : batch, one_hot_pl : action_list, y_ : discounted_loss})
            print("advantage", a)
            print("action", ac)
            print("DEBUG", DEBUG)
        #print("Advantage:",ad)
        #L  = sess.run(unsumm_loss,  feed_dict = {_inputs : batch, y_ : discounted_loss, one_hot_pl : action_list})
        sess.run(optimizer,  feed_dict = {_inputs : batch, y_ : discounted_loss, one_hot_pl : action_list})
        #print("REAL LOSS", L)
    
    makeFolder(savefolder + "final\\")
    saver0.save(sess, savefolder + "best/model.ckpt",  global_step = epoch_i +1 , write_meta_graph= False)
    
    print(start_time)
    end_time  = str(datetime.now())
    print(end_time)

 