import imageio
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

IMG = imageio.imread("./lena.png") / 255.0
GT = imageio.imread("./lena_edges.png") / 255.0

def model(input_img):
	"""Function to predict the edges of Lena.
	Input: 1x512x512x3 dimensional image of Lena. [dtype: tf.float32]
	Output 1x512x512x1 dimensional gray scale image of Lena [dtype: tf.float32]
	"""
	# Number of filters variations through the different layers ensures that we return at the end to a depth of one again.
	# Kernel size was chosen to be 5, as we found it was found to be more optimum and giving better results than 3.
	# Number of strides was chosen to be kept the same, so as not to change the original dimensions of the 512x512 of the picture.
	# After many trials, this number of layers was found to be optimum as more layers led to huge computation time that lasted for so long
 	# And it matched with the number of filters as to give space for it to return back to depth of 1.
	firstL = tf.layers.conv2d(input_img, filters=2, kernel_size=5, strides=1, padding='same', name="firstL")
	secondL = tf.layers.conv2d(firstL, filters=4, kernel_size=5, strides=1, padding='same', name="secondL")
	thirdL = tf.layers.conv2d(secondL, filters=8, kernel_size=5, strides=1, padding='same', name="thirdL")

	fourthL = tf.layers.conv2d(thirdL, filters=16, kernel_size=5, strides=1, padding='same', name="fourthL")
	
	fifthL = tf.layers.conv2d(fourthL, filters=8, kernel_size=5, strides=1, padding='same', name="fifthL")
	sixthL = tf.layers.conv2d(fifthL, filters=4, kernel_size=5, strides=1, padding='same', name="sixthL")
	seventhL = tf.layers.conv2d(sixthL, filters=2, kernel_size=3, strides=1, padding='same', name="seventhL")

	output_tensor_op = tf.layers.conv2d(seventhL, filters=1, kernel_size=3, strides=1, padding='same', name="conv")
	

	return output_tensor_op

def train(N=10000):
	with tf.device('/cpu:0'): # you can use the CPU if you want
		with tf.Graph().as_default():

			input_tensor = tf.constant(np.expand_dims(IMG, axis=0), dtype=tf.float32, shape=[1,512,512,3], name="input")
			gt_tensor = tf.constant(np.expand_dims(np.expand_dims(GT, axis=0), axis=3), dtype=tf.float32, shape=[1,512,512,1], name="gt")

			edge_tensor= model(input_tensor)

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
	with tf.device('/cpu:0'): # you can use the CPU if you want
		with tf.Graph().as_default():

			input_tensor = tf.constant(np.expand_dims(IMG, axis=0), dtype=tf.float32, shape=[1,512,512,3], name="input")			
			edge_tensor = model(input_tensor)
			
			gpu_options = tf.GPUOptions(allow_growth=True)
			sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
			
			saver = tf.train.Saver()
			latest_checkpoint = tf.train.latest_checkpoint("./")
			saver.restore(sess, latest_checkpoint)
	
			#initializing a new session to run the code responsible for displaying the input picture and the output picture
			with sess.as_default():
				sess.run(tf.global_variables_initializer())
				# Printing input image values
				print("Input tensor original")
				print(input_tensor.eval())
				
				# Show input image
				plt.imshow(tf.reshape(input_tensor,[512,512,-1]).eval())
				plt.show()
				
				# Printing output Image values 
				print("Output tensor")
				print(edge_tensor.eval())

				# For the next part, I aim to intrepolate the values from the given output and map them to the range [0,1]
				maxValInTensor = tf.reduce_max(edge_tensor).eval() # getting the max value from the output tensor
				minValInTensor = tf.reduce_min(edge_tensor).eval() # getting the min value from the output tensor
				print(tf.reduce_max(edge_tensor).eval()) 
				print(tf.reduce_min(edge_tensor).eval())
				# Mapping function to call the interpolation function on all elements of the output tensor
				afterInterpolation = tf.map_fn(lambda x: 0 + ((maxValInTensor - minValInTensor) / float((2)) * (x - minValInTensor))
				, edge_tensor)
				print("Interpolation")
				print(afterInterpolation.eval()) # printing values of the tensor after interpolation
				# Showing the image after interpolation in grayscale as the image was getting showed in green at first
				plt.imshow(tf.reshape(afterInterpolation,[512,512]).eval(),cmap='Greys')
				plt.show()

				maxAfterInterpolation =  tf.reduce_max(afterInterpolation).eval() # getting the max value from the interpolated tensor
				minAfterInterpolation =  tf.reduce_min(afterInterpolation).eval() # getting the min value from the interpolated tensor
				# getting the average value to be the thresholding value
				midPoint = minAfterInterpolation + (maxAfterInterpolation - minAfterInterpolation)/2.0

				# Thresholding the values to become binary either black or white in this case
				edgeTmp = tf.reshape(afterInterpolation,[512,512]).eval()
				t = midPoint
				edgeAfterThreshold = (edgeTmp >= t) * 1
				plt.imshow(edgeAfterThreshold,cmap = 'Greys')
				plt.show()

				


			prediction = sess.run(edge_tensor)[0,:,:,0]
			mean_absolute_error = np.mean(np.abs(prediction-GT))
			
			print("Mean Absolute Error: %.4f" % mean_absolute_error)
			return prediction

def main():
	eval()
	# train(10000)

if __name__ == '__main__':
	main()