import tensorflow as tf
from config import return_config_value
from dtype import *
import numpy as np
import os
from util import *

#create config params to store network params?
#static params kept for easy reference and editing, can be made dynamic by calling util


# tf.Session(config = tf.ConfigProto(log_device_replacement=True))

class CNN:
	def __init__(self,save_path,padding = 'VALID'):
		self.graph = tf.Graph()
		self.save_path = save_path

		#load hyperparams from config.ini
		self.img_size 		= return_config_value('HYPERPARAMETERS','img_size',dtype=Integer)
		self.num_channels	= return_config_value('HYPERPARAMETERS','num_channels',dtype=Integer)
		self.pooling_scheme = return_config_value('HYPERPARAMETERS','pooling_scheme',dtype=ListInteger)
		self.dense_scheme	= return_config_value('HYPERPARAMETERS','dense_scheme',dtype=ListInteger)
		self.conv_windows 	= return_config_value('HYPERPARAMETERS','conv_windows',dtype=ListInteger)
		self.pool_windows	= return_config_value('HYPERPARAMETERS','pool_windows',dtype=ListInteger)
		
		
		#learn rate params
		self.learn_rate 	= return_config_value('LEARNING_RATE','learning_rate',dtype=Float)
		self.decay_period	= return_config_value('LEARNING_RATE','decay_period',dtype=Float)
		self.decay_rate		= return_config_value('LEARNING_RATE','decay_rate',dtype=Float)
		self.staircase		= return_config_value('LEARNING_RATE','staircase',dtype=Boolean)
				
		#coherence check
		if len(self.pooling_scheme) != (len(self.pool_windows) + len(self.conv_windows)):
			raise Exception('pooling scheme({}) does not match pool({}) and conv windows({}), check config.ini!'.format(len(self.pooling_scheme),len(self.pool_windows),len(self.conv_windows)))
			
		#counter determining number of layers in model
		self.pool_layers = 0
		self.conv_layers = 0
		self.dense_layers= 0
		self.layers = []
		self.weights = []
		#logging
		print('{} layers detected'.format(len(self.conv_windows)))
		
		with tf.device('/device:GPU:0'):
			
			def convolution(input,
							num_input_filters,
							num_filters,
							window_size,
							activation,
							stride=1,
							padding='SAME',
							customName = None):
				
				self.conv_layers +=1
				stride_window = [1,stride,stride,1]
				
				if customName:
					layer_name = '{}_{}_{}'.format(customName,num_filters,self.conv_layers)
				else:
					layer_name = 'conv_{}_{}'.format(num_filters,self.conv_layers)
				
				w = tf.Variable(tf.truncated_normal([window_size,window_size,num_input_filters,num_filters],
														stddev = 1.0/np.sqrt(num_filters*window_size*window_size),
														name = '{}_w'.format(layer_name)))
														
				b = tf.Variable(tf.zeros(num_filters),name = '{}_b'.format(layer_name))
				
				conv = tf.nn.conv2d(input, 
									filter=w,
									strides=stride_window,
									padding=padding,
									name = layer_name) + b
				
				#TODO
				#apply normalization?
				syn_output = activation(conv)
				
				#keep variable names
				self.layers.append(layer_name)
				self.weights.append('{}_w'.format(layer_name))
				
				return syn_output,num_filters,layer_name
				
			def pooling(input,
						window_size,
						stride=1,
						padding='SAME',
						customName = None):
						
				self.pool_layers+=1
				
				
				if customName:
					layer_name = '{}_{}_{}'.format(customName,num_filters,self.conv_layers)
				else:
					layer_name = 'pool_{}_{}'.format(window_size,self.pool_layers)
				
				window = [1,window_size,window_size,1]
				stride_window = [1,stride,stride,1]
				
				pool = tf.nn.max_pool(value = input,
										ksize = window,
										strides = stride_window,
										padding = padding,
										name = layer_name)
										
										
										
				#keep variable names
				self.layers.append(layer_name)
				
				return pool,layer_name
				
			def dense(input,
						input_dim,
						num_neurons,
						activation,
						customName = None):
						
				self.dense_layers += 1
				
				if customName:
					layer_name = '{}_{}_{}'.format(customName,num_filters,self.conv_layers)
				else:
					layer_name = 'dense_{}_{}'.format(num_neurons,self.dense_layers)
				
				w = tf.Variable(tf.truncated_normal([input_dim,num_neurons],
													stddev = 1.0/np.sqrt(input_dim),
													name = '{}_w'.format(layer_name)))
													
				b = tf.Variable(tf.zeros([num_neurons],
										name = '{}_b'.format(layer_name)))
										
				dense = tf.matmul(input,w) + b
				syn_output = activation(dense)
				
				#keep variable names
				self.layers.append(layer_name)
				self.weights.append('{}_w'.format(layer_name))
				
				return syn_output,num_neurons,layer_name
				
			#linear activation
			def linear(x):
				return x
			
			with self.graph.as_default():
			
				global_step = tf.Variable(0)
				self.learning_rate = tf.train.exponential_decay(self.learn_rate, 
																global_step, 
																decay_steps = self.decay_period, 
																decay_rate = self.decay_rate, 
																staircase = self.staircase)
																
				self.train_image = tf.placeholder(tf.float32, shape=[1,self.img_size*self.img_size*self.num_channels])
				self.train_label = tf.placeholder(tf.float32, shape=[1,len(list(CLASS))])
				
				input_image = tf.reshape(self.train_image, [-1, self.img_size, self.img_size, self.num_channels])
				input_vector = None
				for l in range(len(self.pooling_scheme)):
					if input_vector is None:
						input_vector = input_image
						prev_num_filters = self.num_channels
						
					
					if self.pooling_scheme[l] == 0:
						input_vector,vector_name = pooling(input = input_vector,
											window_size = self.pool_windows[self.pool_layers],
											stride = 1,
											padding = "VALID")

						
					elif self.pooling_scheme[l] > 0:
						input_vector,prev_num_filters,vector_name = convolution(input = input_vector,
																	num_input_filters = prev_num_filters,
																	num_filters = self.pooling_scheme[l],
																	window_size = self.conv_windows[self.conv_layers],
																	activation = tf.nn.relu,
																	stride = 1,
																	padding = "VALID")

						self.weights.append('{}_w'.format(vector_name))
						
					else:
						raise RuntimeError('pooling scheme value {} at layer {] is not valid'.format(pooling_scheme[l],l))
					print(self.layers[-1],input_vector.shape)
					# input_vector,prev_num_filters = convolution(input = input_vector,num_input_filters = prev_num_filters, window_size = conv_windows[l],activation = (tf.nn.relu),stride = 2, padding = 'VALID')
					
				#check layers constructed properly
				if (self.pool_layers+self.conv_layers) != len(self.pooling_scheme):
					raise RunTimeWarning('{} convolution layers and {} pooling layers were added, but there are {} layers in pooling scheme. check config.ini!'.format(conv_iter,pool_iter,len(self.pooling_scheme)))
					
				
					
				#flatten filters and add dense layer(s)
				
				dim = input_vector.get_shape()[1].value * input_vector.get_shape()[2].value * input_vector.get_shape()[3].value 
				conv_out = tf.reshape(input_vector, [-1, dim]) #flatten filters into single image
				
				input_vector = None
				for l in range(len(self.dense_scheme)):
					if input_vector is None:
						input_vector = conv_out
						prev_num_filters = dim
						
					input_vector,prev_num_filters, layer = dense(input_vector,
																input_dim = prev_num_filters,
																num_neurons = self.dense_scheme[l],
																activation = tf.nn.relu)
						    

				#iterate layer names
				for layer in self.layers:
					print(layer)
					
				#output layer
				self.logits,_,vector_name = dense(input_vector,
										input_dim = prev_num_filters,
										num_neurons = len(list(CLASS)),
										activation = linear)
									
				#post-processing
				self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
				
				self.prediction = tf.argmax(self.logits,1)
				self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(self.logits,self.train_label)
				self.loss = tf.reduce_mean(self.cross_entropy)
				self.train_step = self.optimizer.minimize(self.loss)
				
				# self.regularizer = self.lambda_ * tf.nn.l2_loss(weight)
				self.saver = tf.train.Saver()
				
if __name__ == '__main__':
	model = CNN(save_path = os.getcwd()+'/graphs/cnn.ckpt',
				padding = 'VALID')