import tensorflow as tf
from config import return_config_value
from util import *

#create config params to store network params?
#static params kept for easy reference and editing, can be made dynamic by calling util


# tf.Session(config = tf.ConfigProto(log_device_replacement=True))

class CNN:
	def __init__(self,save_path,padding = 'VALID'):
		self.graph = tf.Graph()
		self.save_path = save_path
		
		#load hyperparams from config.ini
		self.img_size 		= return_config_value('HYPERPARMATERS','img_size')
		self.num_channels	= return_config_value('HYPERPARMATERS','num_channels')
		self.pooling_scheme = return_config_value('HYPERPARMATERS','pooling_scheme')
		self.dense_scheme	= return_config_value('HYPERPARMATERS','dense_scheme')
		self.conv_windows 	= return_config_value('HYPERPARMATERS','conv_windows')
		self.pool_windows	= return_config_value('HYPERPARMATERS','pool_windows')
		
		#coherence check
		if len(self.pooling_scheme) != (len(self.pool_windows) + len(self.conv_windows)):
			raise Exception('pooling scheme does not match pool and conv windows, check config.ini!')
			
		#TODO shift to config params
		
		#counter determining number of layers in model
		self.pool_layers = 0
		self.conv_layers = 0
		self.dense_layers= 0
		
		#logging
		print('{} layers detected'.format(len(self.conv_windows)))
		
		with tf.device('/device:GPU:0'):
			
			def convolution(input,
							num_input_filters,
							num_filters,
							window_size,
							activation,
							stride=1,
							padding='SAME'):
				
				self.conv_layers +=1
				stride_window = [1,stride,stride,1]
				layer_name = 'conv_{}_{}'.format(num_filters,self.conv_layers)
				
				w = tf.Variable(tf.truncated_normal([window_size,window_size,num_input_filters,num_filters],
														stddev = 1.0/np.sqrt(num_filters*window_size*window_size),
														name = '{}_w'.format(layer_name)))
														
				b = tf.Variable(tf.zeros(num_filters),name = '{}_b'.format(layer_name))
				
				conv = tf.nn.convo2d(input, 
									filter=w,
									strides=stride_window,
									padding=padding,
									name = layer_name) + b
				
				#TODO
				#apply normalization?
				syn_output = activation(conv)
				
				return syn_output,num_filters,layer_name
				
			def pooling(input,
						window_size,
						stride,
						padding):
						
				self.pool_layers+=1
				stride_window = [1,stride,stride,1]
				layer_name = 'pool_{}_{}'.format(window_size,self.pool_layers)
				
				pool = tf.nn.max_pool(value = input,
										ksize = window_size,
										strides = stride_window,
										padding = padding,
										name = layer_name)
										
				return pool,layer_name
				
			def dense(input,
						input_dim,
						num_neurons,
						activation):
				layer_name = 'dense_{}_{}'.format(num_neurons,self.dense_layers)
				
				w = tf.Variable(tf.truncated_normal([input_dim,num_neurons],
													stddev = 1.0/np.sqrt(input_dim),
													name = '{}_w'.format(layer_name)))
													
				b = tf.Variable(tf.zeros([input_dim,num_neurons],
										name = '{}_w'.format(layer_name)))
										
				dense = tf.matmul(input,w) + b_fc
				syn_output = activation(dense)
				
				return syn_output,num_neurons,layer_name
				
				#linear activation
				def linear(x):
					return x
			
			with self.graph_as_default():
				self.train_image = tf.placeholder(tf.float32, shape=[1,self.img_size*self.img_size*self.num_channels])
				self.train_label = tf.placeholder(tf.float32, shape=[1,len(list(CLASSES))])
				

				conv_iter,pool_iter = 0,0
				layers = []
				
				input_vector = None
				for l in range(len(self.pooling_scheme)):
					if input_vector is None:
						input_vector = self.train_image
						prev_num_filters = self.num_channels
						
					if pooling_scheme[l] == 1:
						input_vector,vector_name = pool(input = input_vector,
											window_size = self.pool_windows[pool_iter],
											stride = 2,
											padding = "VALID")
						pool_iter+=1
						layers.append(vector_name)
						
					elif pooling_scheme[l] == 0:
						input_vector,prev_num_filters,vector_name = convolution(input = input_vector,
																	num_input_filters = prev_num_filters,
																	window_size = self.conv_window[conv_iter],
																	activation = tf.nn.relu,
																	stride = 2,
																	padding = "VALID")
						conv_iter+=1
						layers.append(vector_name)
						
					else:
						raise RuntimeError('pooling scheme value {} at layer {] is not valid'.format(pooling_scheme[l],l))
					
					input_vector,prev_num_filters = convolution(input = input_vector,num_input_filters = prev_num_filters, window_size = conv_windows[l],activation = (tf.nn.relu),stride = 2, padding = 'VALID')
					
				#check layers constructed properly
				if (pool_iter+conv_iter) != len(self.pooling_scheme):
					raise RunTimeWarning('{} convolution layers and {} pooling layers were added, but there are {} layers in pooling scheme. check config.ini!'.format(conv_iter,pool_iter,len(self.pooling_scheme)))
					
				
					
				#flatten filters and add dense layer(s)
				
				dim = input_vector.get_shape()[1].value * input_vector.get_shape()[2].value * input_vector.get_shape()[3].value 
				conv_out = tf.reshape(input_vector, [-1, dim]) #flatten filters into single image
				
				input_vector = None
				for i in range(len(dense_scheme)):
					if input_vector = None:
						input_vector = conv_out
						prev_num_filters = dim
						
					
					input_vector,prev_num_filters, layer = dense(input_vector,
								input_dim = prev_num_filters,
								num_neurons = self.dense_scheme[i],
								acivation = tf.nn.relu)
						    
					layers.append(layer)

				#iterate layer names
				for layer in layers:
					print(layer)
					
				#output layer
				self.logits = tf.dense(input_vector,
										input_dim = prev_num_filters,
										num_neurons = len(list(CLASSES)),
										activation = linear)
									
				
				#post-processing
				self.prediction = 
				self.train_prediction = self.logits
				self.saver = tf.train.Saver()