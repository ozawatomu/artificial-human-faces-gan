import tensorflow.compat.v1 as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from image import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generator(noise, reuse=None):
	with tf.variable_scope('gen', reuse=reuse):
		hidden1 = tf.layers.dense(inputs=noise, units=256)
		hidden1 = tf.nn.swish(hidden1)

		hidden2 = tf.layers.dense(inputs=hidden1, units=1024)
		hidden2 = tf.nn.swish(hidden2)

		output = tf.layers.dense(hidden2, units=4096, activation=tf.nn.tanh)
		return output

noise = tf.placeholder(tf.float32, shape=[None, 128])


samples = []
saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess, 'models/model_1.ckpt')
	for i in range(5):
		samples_noise = np.random.uniform(-1, 1, size=(1, 128))
		gen_sample = sess.run(generator(noise, reuse=True), {noise: samples_noise})
		samples.append(gen_sample)

plt.figure(0)
plt.imshow(samples[0].reshape(64, 64), cmap='gray')
plt.figure(1)
plt.imshow(samples[1].reshape(64, 64), cmap='gray')
plt.figure(2)
plt.imshow(samples[2].reshape(64, 64), cmap='gray')
plt.figure(3)
plt.imshow(samples[3].reshape(64, 64), cmap='gray')
plt.figure(4)
plt.imshow(samples[4].reshape(64, 64), cmap='gray')
plt.show()