from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf



class logistic(object):
	def __init__(self,mnist):
	
		self.mnist=mnist
		self.x=tf.placeholder(tf.float32,shape=[None,784])
		self.T=tf.placeholder(tf.float32,shape=[None,10])
		self.W=tf.Variable(tf.zeros([784,10]))
		self.b=tf.Variable(tf.zeros([10]))


	def fit(self):
		y=self.forward(self.x)
		cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.T,logits=y))
		train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for _ in range(1000):
			    batch = self.mnist.train.next_batch(100)
			    sess.run(train_step, feed_dict={self.x: batch[0], self.T: batch[1]})
			correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self.T,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			print(accuracy.eval(feed_dict={self.x: self.mnist.test.images, self.T: self.mnist.test.labels}))


	def predict(self,x):
		act=self.forward(x)
		return tf.argmax(act,1)


	def forward(self,x):
		return tf.matmul(x,self.W) + self.b



mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
model=logistic(mnist)
model.fit()







