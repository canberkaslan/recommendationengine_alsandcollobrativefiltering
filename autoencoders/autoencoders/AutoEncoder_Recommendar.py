# Importing tensorflow
import data as data
import tensorflow as tf
# Importing some more libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from surprise import Reader, Dataset,  evaluate
from surprise.model_selection import cross_validate, train_test_split
from timeit import default_timer as timer
# reading the ratings data

header = ['user_id', 'item_id', 'rating', 'timestamp']
#df = pd.read_csv('D:\\Ozyegin\\2018-Guz\\CS556_BigDataAnalysis\\ratings.dat', sep='::',header = None, names =header, usecols = [0,1,2,3])
df = pd.read_csv('D:\\Ozyegin\\2018-Guz\\CS556_BigDataAnalysis\\u.data', sep='\t',header = None, names =header, usecols = [0,1,2,3])
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

M = np.zeros((n_users,n_items))
M[df.user_id.values-1,df.item_id.values-1]=df.rating.values


def split_test_train(R, p=0.25):
    I = np.random.random(size=R.shape)
    test = R*(I<p).astype(float)
    train= R*(I>=p).astype(float)
    return train,test


def var(name, shape, std=None):
    if not std:
        std = (1/shape[0])**0.5
    init = tf.truncated_normal_initializer(stddev=std)
    return tf.get_variable(name=name, dtype=tf.float64,
                           shape=shape, initializer=init)

trainM, testM = split_test_train(M,0.2)
print(testM.item(99))
print("Split data")
#rowSize=n_items
n_movies = n_items
'''
g = tf.Graph()
with g.as_default():
    X = tf.placeholder(dtype=tf.float64, shape=[None, n_movies])
    mask_train = tf.constant((trainM > 0).astype(float), dtype=tf.float64)
    mask_test = tf.constant((testM > 0).astype(float), dtype=tf.float64)
    keep_prob = tf.placeholder(dtype=tf.float64)

    # encoder
    W1 = var('W1', [n_movies, 500])
    b1 = var('b1', [500])
    out1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    out1 = tf.nn.dropout(out1, keep_prob)

    W2 = var('W2', [500, 100])
    b2 = var('b2', [100])
    encoded = tf.nn.relu(tf.matmul(out1, W2) + b2)
    encoded = tf.nn.dropout(encoded, keep_prob)

    # decoder
    W3 = var('W3', [100, 500])
    b3 = var('b3', [500])
    out3 = tf.nn.relu(tf.matmul(encoded, W3) + b3)
    out3 = tf.nn.dropout(out3, keep_prob)

    W4 = var('W4', [500, n_movies])
    b4 = var('b4', [n_movies])
    decoded = tf.matmul(out3, W4) + b4

    loss = tf.reduce_sum(((X - decoded) * mask_train) ** 2) \
           / tf.reduce_sum(mask_train)

    rmse_train = loss ** 0.5
    rmse_test = (tf.reduce_sum(((X - decoded) * mask_test) ** 2) \
                 / tf.reduce_sum(mask_test)) ** 0.5
    step = tf.train.AdamOptimizer().minimize(loss)
'''
g = tf.Graph()
with g.as_default():
    X = tf.placeholder(dtype=tf.float64, shape=[None, n_movies])
    mask_train = tf.constant((trainM > 0).astype(float), dtype=tf.float64)
    mask_test = tf.constant((testM > 0).astype(float), dtype=tf.float64)

    keep_prob = tf.placeholder(dtype=tf.float64)
    #Encoder
    W1 = var('W1', [n_movies, 10])
    b1 = var('b1', [10], 0.0001)
    encoded = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    encoded = tf.nn.dropout(encoded, keep_prob)
    
    #Decoder
    W2 = var('W2', [10, n_movies])
    b2 = var('b2', [n_movies])
    decoded = tf.matmul(encoded, W2) + b2

    loss = tf.reduce_sum(((X - decoded) * mask_train) ** 2) \
           / tf.reduce_sum(mask_train)

    rmse_train = loss ** 0.5
    rmse_test = (tf.reduce_sum(((X - decoded) * mask_test) ** 2) \
                 / tf.reduce_sum(mask_test)) ** 0.5

    step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.InteractiveSession(graph=g)
tf.global_variables_initializer().run()

n_users = 943
batch_size = 943
n_batches = n_users // batch_size

start = timer()

for epoch in range(100):
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        sess.run(step, feed_dict={X: trainM[start:end], keep_prob: 0.5})
        print('iter: %d/%d' % (i, n_batches), end='\r')

    a, b = sess.run([rmse_train, rmse_test], feed_dict={X: M, keep_prob: 1.})
    print('epoch:%d \t rmse(train):%1.4f \t rmse(test):%1.4f' % (epoch, a, b))

duration = timer() - start
print("_______________________________________________")
print("Time for Duration:%2.4f",duration)