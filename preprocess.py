'''
CLASS SPECIFICATION
1.	Black Grass
2.	Charlock
3.	Cleavers
4.	Common Chickweed
5.	Common Wheat
6.	Fat Hen
7.	Loose Silky-Bent
8.	Maize
9.	Scentless Mayweed
10.	Shepherds Purse
11.	Small Flowered Cranesbill
12.	Sugar Beet
'''

import os
import pickle
import imageio
from numpy import array,reshape,zeros,uint8,arange
from tensorflow import one_hot
import matplotlib as plt
from PIL import Image
from skimage.transform import resize as imresize

from config import *
from util import *
from dtype import Integer
train_path_img = os.getcwd() + '/data/train/img/'
train_path_thumb = os.getcwd() + '/data/train/thumb/'
test_path_img = os.getcwd() + '/data/test/img/'
test_path_thumb = os.getcwd() + '/data/test/thumb/'

	

def preprocess():
	print('preprocess')
	create_dummy_files()
	convert_to_thumbnail()
	flag('_init_run','no')
	
#dummy testing file
def dummy():
	# img = imageio.imread(os.getcwd()+'/data/train/img/Black-grass/0ace21089.png')
	img = Image.open(os.getcwd()+'/data/train/img/Black-grass/418808d19.png')
	img.resize((128,128),Image.BICUBIC).save('test.png')
	# img.transform(size=(128,128),method=Image.BICUBIC,data=Image.BICUBIC)
	# img.save('test.png')
	# print(img.shape)
	# img = imresize(img,(51,51,3))
	# imageio.imsave('test.png',img)
	
def create_dummy_files():
	dummy = array([[[0,0,0],[1,1,1]]]).astype(uint8)
	for _,grass_type in CLASS.items():
		for file in os.listdir(train_path_img+grass_type):
			fn = train_path_thumb+grass_type+'/'+file
			with open(fn,'w') as file:
				file.write('1')
			imageio.imsave(fn,dummy)
	for file in os.listdir(test_path_img):
		fn = test_path_thumb+'/'+file
		with open(fn,'w') as file:
			file.write('1')
		imageio.imsave(fn,dummy)
		
def convert_to_thumbnail():	

	resize_dim = return_config_value('HYPERPARAMETERS','img_size',dtype=Integer)
	resize_dim = config['HYPERPARAMETERS']['img_size']
	#train images
	for _,grass_type in CLASS.items():
		for file in os.listdir(train_path_img+grass_type):
			img = Image.open(train_path_img+grass_type+'/'+file)
			img.resize((resize_dim,resize_dim),Image.BICUBIC).save(train_path_thumb+grass_type+'/'+file)
			
	#test images
	for file in os.listdir(test_path_thumb):
		img = Image.open(test_path_img+'/'+file)
		img.resize((resize_dim,resize_dim),Image.BICUBIC).save(test_path_thumb+'/'+file)

#load train images into array
def load_train_images():
	x,y = [],[]
	for _,grass_type in CLASS.items():
		for file in os.listdir(train_path_thumb+grass_type):
			img = imageio.imread(train_path_thumb+grass_type+'/'+file)
			x.append(img)
			y.append(grass_type)
			
	y_ = zeros((len(x),len(list(CLASS))))
	for i in range(len(y)):
		y_[i,CLASS_INVERSE[y[i]]-1] = 1
		
	return x,y_
	
# trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
# trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix
	
def load_test_images():
	y = []
	for file in os.listdir(test_path_thumb):
		img = imageio.imread(test_path_thumb+'/'+file)
		y.append(img)
		
	return y
	
if __name__ == '__main__':
	preprocess()