import imageio
import tensorflow as tf
import numpy as np

IMG = imageio.imread("./lena.png") / 255.0
GT = imageio.imread("./lena_edges.png") / 255.0


def model(input_img):
	"""Function to predict the edges of Lena.
	Input: 1x512x512x3 dimensional image of Lena. [dtype: tf.float32]
	Output 1x512x512x1 dimensional gray scale image of Lena [dtype: tf.float32]
	"""
	
	# Please replace the next line with your custom code and add some describing comments
	output_tensor_op = tf.layers.conv2d(input_img, filters=1, kernel_size=3, strides=1, padding='same', name="conv")
	
	return output_tensor_op
	

def train(N=10000):
	with tf.device('/gpu:0'): # you can use the CPU if you want
		with tf.Graph().as_default():

			input_tensor = tf.constant(np.expand_dims(IMG, axis=0), dtype=tf.float32, shape=[1,512,512,3], name="input")
			gt_tensor = tf.constant(np.expand_dims(np.expand_dims(GT, axis=0), axis=3), dtype=tf.float32, shape=[1,512,512,1], name="gt")

			edge_tensor = model(input_tensor)

			loss = tf.reduce_mean(tf.abs(tf.subtract(edge_tensor, gt_tensor))) # average absolute error

			optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
			global_step = tf.Variable(0, name='global_step', trainable=False)
			train_op = optimizer.minimize(loss, global_step=global_step)

			gpu_options = tf.GPUOptions(allow_growth=True)
			sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
			sess.run(tf.global_variables_initializer())

			for i in range(1,N+1):
				_, loss_val = sess.run([train_op, loss])
				if i % 10 == 0:
					print("Step: %d \tLoss: %f" % (i,loss_val))
					
			saver = tf.train.Saver(max_to_keep=1)
			saver.save(sess, "./ckpt", global_step=i)
			return
			

def eval():
	with tf.device('/gpu:0'): # you can use the CPU if you want
		with tf.Graph().as_default():

			input_tensor = tf.constant(np.expand_dims(IMG, axis=0), dtype=tf.float32, shape=[1,512,512,3], name="input")			
			edge_tensor = model(input_tensor)
			
			gpu_options = tf.GPUOptions(allow_growth=True)
			sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
			
			saver = tf.train.Saver()
			latest_checkpoint = tf.train.latest_checkpoint("./")
			saver.restore(sess, latest_checkpoint)
	
			prediction = sess.run(edge_tensor)[0,:,:,0]
			mean_absolute_error = np.mean(np.abs(prediction-GT))
			
			print("Mean Absolute Error: %.4f" % mean_absolute_error)
			return prediction