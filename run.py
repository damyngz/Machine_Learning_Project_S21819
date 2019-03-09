import preprocess
import os
import model
from config import *

def run():
	#load config params to decide any pre-run actions needed
	prerun_check()
	
	a = model.CNN(save_path = os.getcwd()+'/graphs/cnn.ckpt',
				padding = 'VALID')
	
if __name__ == '__main__':
	run()