CLASS = {
1:	'Black-grass',
2:	'Charlock',
3:	'Cleavers',
4:	'Common Chickweed',
5:	'Common wheat',
6:	'Fat Hen',
7:	'Loose Silky-bent',
8:	'Maize',
9:	'Scentless Mayweed',
10:	"Shepherds Purse",
11:	'Small-flowered Cranesbill',
12: 'Sugar beet'
}

CLASS_INVERSE = {}
for class_,type_ in CLASS.items():
	CLASS_INVERSE[type_] = class_

#default config params

DEFAULT = {
'_init_run' : 'no'
}

#default config hyperparam section
CFG_DEFAULT_HYPERPARAM = {
'pooling_scheme':	[64,128,0,128,156,0,156,192,0,192,256,0,256,0],
'dense_scheme'	:	[256,128],	
'pool_windows'	:	[3,3,3,3,3],
'conv_windows' 	: 	[3,3,3,3,3,3,3,3,3],
'img_size' 		:	128,
'num_channels'	:	3,
'beta'			:	0.0001,
'num_epochs'	:	10,
'batch_size'	:	1,
'seed'			:	10
}

CFG_DEFAULT_HYPERPARAM_LEARNING_RATE = {
'learning_rate'	:	0.01,
'decay_period'	:	1000,
'decay_rate'	:	0.95,
'staircase'		:	True
}


CFG_DEFAULT_BUILD = {'DEFAULT' : DEFAULT,
			'HYPERPARAMETERS' : CFG_DEFAULT_HYPERPARAM,
			'LEARNING_RATE' : CFG_DEFAULT_HYPERPARAM_LEARNING_RATE
			}
