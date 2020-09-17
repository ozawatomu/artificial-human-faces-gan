import tensorflow.compat.v1 as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from image import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SETUP DATA
images = []
dir_path = 'src/training_data'
for file in os.listdir(dir_path):
	images.append(Image('{}/{}'.format(dir_path, file)).get_as_numpy_array().flatten()/255)
# print(images[0])
# plt.figure(1)
# plt.imshow(images[4].reshape(64, 64), cmap='gray')
# plt.figure(2)
# plt.imshow(images[6].reshape(64, 64), cmap='gray')
# plt.figure(3)
# plt.imshow(images[8].reshape(64, 64), cmap='gray')
# plt.show()




def generator(noise, reuse=None):
	with tf.variable_scope('gen', reuse=reuse):
		hidden1 = tf.layers.dense(inputs=noise, units=256)
		hidden1 = tf.nn.swish(hidden1)

		hidden2 = tf.layers.dense(inputs=hidden1, units=1024)
		hidden2 = tf.nn.swish(hidden2)

		output = tf.layers.dense(hidden2, units=4096, activation=tf.nn.tanh)
		return output


def discriminator(image, reuse=None):
	with tf.variable_scope('dis', reuse=reuse):
		hidden1 = tf.layers.dense(inputs=image, units=1024)
		hidden1 = tf.nn.swish(hidden1)

		hidden2 = tf.layers.dense(inputs=hidden1, units=256)
		hidden2 = tf.nn.swish(hidden2)

		logits = tf.layers.dense(inputs=hidden2, units=1)
		output = tf.sigmoid(logits)

		return output, logits


real_images = tf.placeholder(tf.float32, shape=[None, 4096])
noise = tf.placeholder(tf.float32, shape=[None, 128])

G = generator(noise)

D_output_real, D_logits_real = discriminator(real_images)

D_output_fake, D_logits_fake = discriminator(G, True)


# LOSSES
def loss_func(logits_in, labels_in):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))


D_real_loss = loss_func(D_logits_real, tf.ones_like(D_logits_real)*0.9)

D_fake_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_fake))

D_loss = D_real_loss + D_fake_loss

G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))


learning_rate = 0.0001

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)


batch_size = 50
epochs = 200

init = tf.global_variables_initializer()
samples_over_epochs = []
samples = []
saver = tf.train.Saver(max_to_keep=100)
# with tf.Session() as sess:
# 	sess.run(init)
# 	for epoch in range(epochs):
# 		print("\rON EPOCH {}".format(epoch), end='')
# 		num_baches = len(images)//batch_size
# 		for i in range(num_baches):
# 			batch_i = i*batch_size
# 			batch = np.array([images[batch_i]])
# 			for j in range(1, batch_size):
# 				batch = np.vstack([batch, images[batch_i + j]])
# 			batch = batch * 2 - 1
# 			batch_noise = np.random.uniform(-1, 1, size=(batch_size, 128))
# 			_ = sess.run(D_trainer, {real_images:batch, noise:batch_noise})
# 			_ = sess.run(G_trainer, {noise:batch_noise})
# 		if epoch % 5 == 0:
# 			saver.save(sess, 'models/model_{}/model.ckpt'.format(epoch))
# 	saver.save(sess, 'models/model_{}/model.ckpt'.format(epochs))

epoch_to_print = 50

with tf.Session() as sess:
	saver.restore(sess, 'models/model_{}/model.ckpt'.format(epoch_to_print))
	for i in range(400):
		samples_noise = np.random.uniform(-1, 1, size=(1, 128))
		gen_sample = sess.run(generator(noise, reuse=True), {noise: samples_noise})
		samples.append(gen_sample)

# with tf.Session() as sess:
# 	samples_noise = np.random.uniform(-1, 1, size=(1, 128))
# 	for i in range(5):
# 		saver.restore(sess, 'models/model_{}/model.ckpt'.format(10*i))
# 		gen_sample = sess.run(generator(noise, reuse=True), {noise: samples_noise})
# 		samples.append(gen_sample)

fig = plt.figure(frameon=False, figsize=(1, 1))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
for i in range(len(samples)):
	ax.imshow(samples[i].reshape(64, 64), cmap='gray')
	fig.savefig('Results/{}-{}'.format(epoch_to_print, i), dpi=256)
# plt.figure(0)
# plt.imshow(samples[0].reshape(64, 64), cmap='gray')
# plt.figure(1)
# plt.imshow(samples[1].reshape(64, 64), cmap='gray')
# plt.figure(2)
# plt.imshow(samples[2].reshape(64, 64), cmap='gray')
# plt.figure(3)
# plt.imshow(samples[3].reshape(64, 64), cmap='gray')
# plt.figure(4)
# plt.imshow(samples[4].reshape(64, 64), cmap='gray')
# plt.show()