# Pradeep Saini

# CSV Data file can not pushed on github due to file size constraint of 100mb
# so here is the link for real time series future data
# https://www.dropbox.com/s/hcrxfu9ztfpbsrw/irage_dataset.csv?dl=0
# Download it and place in working directory

import math
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

x = tf.placeholder('float', [None, 142])
y = tf.placeholder('float')

# class NeuralNetwork():
def test_train_data():
    shuffled_df = df.sample(frac =1)
    classifier_column = "fut_direction"
    temp_input = shuffled_df[0:len(shuffled_df)]       
    temp_input.fillna(value = -99999, inplace=True)
    temp_input_x = temp_input.drop(["date", "fut_spread", "fut_direction", "30secAhead", "1minAhead"], 1)
    temp_output_y_arr = np.array(temp_input[classifier_column])
    labels = []
    for i in range(len(temp_output_y_arr)):
        arr = [0, 0, 0, 0, 0]
        arr[int(temp_output_y_arr[i]) + 2 ] += 1
        labels.append(arr)

    temp_x = preprocessing.scale(temp_input_x)
#     temp_x = temp_input_x
    x_train, x_test, y_train, y_test = cross_validation.train_test_split((temp_x), np.array(labels), test_size=0.10)
    train_data = []
    for i in range(len(x_train)):
        arr = []
        arr.append(list(x_train[i]))
        arr.append(list(y_train[i]))
        train_data.append(arr)
    return np.array(train_data), x_test, y_test

def formated_np_batch(arr, start, end):
    new_arr = []
    for i in range(start, end):
        if(i == len(arr)):
            break
        new_arr.append(arr[i])
    new_np_arr =  np.array(new_arr)
    return new_np_arr

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([142, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

#     hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
#                       'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
#     hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
#                       'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
    
#     hidden_5_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
#                       'biases':tf.Variable(tf.random_normal([n_nodes_hl5]))}
    
    
#     hidden_6_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl5, n_nodes_hl6])),
#                       'biases':tf.Variable(tf.random_normal([n_nodes_hl6]))}

#     output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
#                     'biases':tf.Variable(tf.random_normal([n_classes])),}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes])),}




    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)

#     l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
#     l3 = tf.nn.sigmoid(l3)
    
#     l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])
#     l4 = tf.nn.relu(l4)
    
#     l5 = tf.add(tf.matmul(l4,hidden_5_layer['weights']), hidden_5_layer['biases'])
#     l5 = tf.nn.relu(l5)
    
#     l6 = tf.add(tf.matmul(l5,hidden_6_layer['weights']), hidden_6_layer['biases'])
#     l6 = tf.nn.relu(l6)

#     output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    output = tf.matmul(l2,output_layer['weights']) + output_layer['biases']
    return output

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_nodes_hl4 = 500
n_classes = 5
batch_size = 5000 # ??
hm_epochs = 10

def train_neural_network(x, xy_train, x_test, y_test):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10
    try:
        tf_log = 'tf.log'
        saver = tf.train.Saver()
    except:
        pass
    with tf.Session() as sess:
        start_time = time.time()
        epoch = 1
        sess.run(tf.initialize_all_variables())
        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2]) + 1
            print ("starting : ", epoch)
        except:
            epoch = 1
        
        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess, "model.ckpt")
            epoch_loss = 1
#             print xy_train
            np.random.shuffle(xy_train)
            i = 0
            while (i < math.ceil(len(xy_train)/(1.0 *batch_size))):
                epoch_x = formated_np_batch(xy_train[:, 0], i * batch_size, (i+1)*batch_size)
                epoch_y = formated_np_batch(xy_train[:, 1], i * batch_size, (i+1)*batch_size)
                _, c =  sess.run([optimizer, cost], feed_dict = {x : epoch_x, y : epoch_y })
                epoch_loss += c
                i = i + 1
            
            saver.save(sess, "model.ckpt")
            print('Epoch', epoch, 'completed out of ', hm_epochs, 'loss: ', epoch_loss)
            with open(tf_log, 'a') as f:
                f.write(str(epoch) + "\n")
            epoch += 1
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))
        end_time = time.time()
        print ("time taken for four hidden layer (500, 500, 500, 500), input:output 0.1, input data = 640000 , batch_size = " + str(batch_size) + ", time taken in secs:: " + str(end_time - start_time))
        result = sess.run(tf.argmax(prediction.eval(feed_dict={x:x_test}),1))
        return result
    
df = pd.read_csv('irage_dataset.csv')
xy_train, x_test, y_test = test_train_data()
print len(xy_train)
print len(x_test)
# print x_test[1:10]
# print y_test[1:10]

def accuracy_matrix(predicted_test_result):
    narr = np.array(y_test)
    y_test_real = [np.argmax(a) for a in narr]
    print confusion_matrix(y_test_real, predicted_test_result)
    print accuracy_score(y_test_real, predicted_test_result)
    
predicted_test_result = train_neural_network(x, xy_train, x_test, y_test)
accuracy_matrix(predicted_test_result)

def use_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,"model.ckpt")
        result = (sess.run(prediction.eval(feed_dict={x:x_test})))
        return result

test_results = use_neural_network()
#class 0 -> future direction -2
#class 1 -> future direction -1
#class 2 -> future direction 0
#class 3 -> future direction 1
#class 4 -> future direction 2
