import tensorflow as tf

#create config params to store network params?
#static params kept for easy reference and editing, can be made dynamic by calling util


# tf.Session(config = tf.ConfigProto(log_device_replacement=True))

class CNN:
	def __init__(self,save_path,padding = 'VALID'):
		self.graph = tf.Graph()
		self.save_path = save_path
		
		#load hyperparams from config.ini
		self.conv_windows 	= config.return_config_value('HYPER','conv_windows')
		self.img_size 		= config.return_config_value('HYPER','img_size')
		self.num_channels	= config.return_config_value('HYPER','num_channels')
		
		#TODO shift to config params
		
		#counter determining number of layers in model
		self.layers = 0
		
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
				
				self.layers +=1
				stride_window = [1,stride,stride,1]
				layer_name = 'conv_{}_{}'.format(num_filters,self.layers)
				w = tf.Variable(tf.truncated_normal([window_size,window_size,num_input_filters,num_filters),
														stddev = 1.0/np.sqrt(num_filters*window_size*window_size),
														name = '{}_w'.format(layer_name))
														
				b = tf.Variable(tf.zeros(num_filters),name = '{}_b'.format(layer_name))
				
				conv = tf.nn.convo2d(input, 
									filter=w,
									strides=stride_window,
									padding=padding) + b
				
				#TODO
				#apply normalization?
				syn_output = activation(conv)
				
				return syn_output,window_size
			
			with self.graph_as_default():
				input_vectors = [0 for i in range(len(conv_windows))]
				input_vectors[0] = self.input
				
				self.image = tf.placeholder(tf.float32,[self.img_size*self.img_size*self.num_channels]
				for l in range(len(conv_windows)):
					if input_vector is None:
						input_vector = self.input_image
						prev_num_filters = self.num_channels
					
					input_vector,prev_num_filters = convolution(input = input_vector,num_input_filters = prev_num_filters, window_size = conv_windows[l],activation = (tf.nn.relu),stride = 2, padding = 'VALID')
					
					#TODO recursion params here
				W_conv1 = tf.Variable(tf.truncated_normal([self.conv_window[0], self.conv_window[0], self.num_channels, self.num_filters], stddev=1.0/np.sqrt(self.num_channels*self.window_size*self.window_size)), name='weights_1')
				b_conv1 = tf.Variable(tf.zeros([NUM_CONV1_FILTERS]), name='biases_1')
				
				W_conv2 = tf.Variable(tf.truncated_normal([5, 5, NUM_CONV1_FILTERS, NUM_CONV2_FILTERS], stddev=1.0/np.sqrt(NUM_CONV1_FILTERS*5*5)), name='weights_2')
				b_conv2 = tf.Variable(tf.zeros([NUM_CONV2_FILTERS]), name='biases_2')
				
				W_fc = tf.Variable(tf.truncated_normal([dim,NUM_FC_NEURONS], stddev=1.0/np.sqrt(dim)), name='weights_3')
				b_fc = tf.Variable(tf.zeros([NUM_FC_NEURONS]), name='biases_3')
				
				W_out = tf.Variable(tf.truncated_normal([NUM_FC_NEURONS, NUM_CLASSES], stddev=1.0/np.sqrt(NUM_FC_NEURONS)), name='weights_4')
				b_out = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
				
				#input layer
				#-1 neccessary?
				images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
				
				#layers
				#TODO 6-9 layers CNN with max_pooling?
				
				conv_1 = tf.nn.relu(tf.nn.conv2d(images, W_conv1, [1, 1, 1, 1], padding='VALID') + b_conv1)
				pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')
				conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W_conv2, [1, 1, 1, 1], padding='VALID') + b_conv2)
				pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')
				dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value 
				pool_2_flat = tf.reshape(pool_2, [-1, dim]) #flatten 3 channels into single channel
				fc_ = tf.nn.relu(tf.matmul(pool_2_flat,W_fc) + b_fc)
				
				#output layer
				logits = tf.matmul(fc_, W_out) + b_out
				
				#saver and predictions
				#[-1] to dim neccessary?
				self.train_prediction = self.logits[-1] 
				self.saver = tf.train.Saver()