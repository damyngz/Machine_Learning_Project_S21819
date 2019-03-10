import preprocess
import os
import model
from config import *

def run():
	#load config params to decide any pre-run actions needed
	prerun_check()
	
	
	cnn = model.CNN()
	x,y = preprocess.load_train_images()
	model.load_model(cnn)
	model.train_model(cnn,x,y)
	
if __name__ == '__main__':
	run()