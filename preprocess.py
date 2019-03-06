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
from numpy import array,reshape,zeros
import matplotlib as plt
from PIL import Image
from util import *
from skimage.transform import resize as imresize

resize_dim = 51
train_path_img = os.getcwd() + '/data/train/img/'
train_path_thumb = os.getcwd() + '/data/train/thumb/'

	
def preprocess():
	# load_train_images()
	convert_to_thumbnail()
	# dummy()
	return
	
def dummy():
	# img = imageio.imread(os.getcwd()+'/data/train/img/Black-grass/0ace21089.png')
	img = Image.open(os.getcwd()+'/data/train/img/Black-grass/418808d19.png')
	img.resize((128,128),Image.BICUBIC).save('test.png')
	# img.transform(size=(128,128),method=Image.BICUBIC,data=Image.BICUBIC)
	# img.save('test.png')
	# print(img.shape)
	# img = imresize(img,(51,51,3))
	# imageio.imsave('test.png',img)
	
	
def convert_to_thumbnail():	
	for _,grass_type in class_.items():
		for file in os.listdir(train_path_img+grass_type):
			img = Image.open(train_path_img+grass_type+'/'+file)
			img.resize((128,128),Image.BICUBIC).save(train_path_thumb+grass_type+'/'+file)
			
def load_train_images():
	x = []
	i = 5
	for _,grass_type in class_.items():
		for file in os.listdir(train_path_img+grass_type):
			img = imageio.imread(train_path_img+grass_type+'/'+file)
			# print(img)
			# print('{} {}'.format(file,img.shape))
			x.append(img)
			i-=1
			if i==0:
				break
				
		break
	imageio.imsave('test.png',im = x[1])
def resize_to_dim():
	return
	
if __name__ == '__main__':
	preprocess()