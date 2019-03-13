import configparser
import os
import logging 
import runpy 
import dtype
from dtype import String

config_path = 'config.ini'
preprocess_dir = 'preprocess.py'

def load_config(force_build = False):
	config = configparser.ConfigParser()
	
	#creates a config file if non exists
	if os.path.isfile(config_path) and not(force_build):
		config.read(config_path)
	else:
		from util import CFG_DEFAULT_BUILD
		for section, params in CFG_DEFAULT_BUILD.items():
			config[section] = params
		print('build_config_file')
		# logging.info('build config file')
		with open(config_path,'w') as file:
			config.write(file)
			
	return config
	
def return_config_value(section,flag,dtype = String):
	global config
	try:
		return dtype(config[section][flag])
	except KeyError:
		print('config values not found')
	
def prerun_check():
	global config
	config = load_config()
	_init_run = config['DEFAULT']['_init_run']
	# print(_init_run)
	if _init_run == 'yes':
		print('building images')
		from preprocess import preprocess
		preprocess()
	elif _init_run != 'no':
		print('config.ini corrupted, re-creating files')	
		load_config(force_build = True)
		print('building images')
		from preprocess import preprocess
		preprocess()
		
	print('pre-run success')
		
def flag(flag_,flag = None):
	try:
		config['DEFAULT'][flag_] = flag
		with open(config_path,'w') as file:
			config.write(file)
	except KeyError:
		raise RuntimeError('config flag {} does not exist.'.format(flag_))