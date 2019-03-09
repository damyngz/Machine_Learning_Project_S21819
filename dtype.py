#for parsing config values	
def Boolean(n):
	if n.lower() == 'true':
		return True
	elif n.lower() == 'false':
		return False
	else:
		raise Exception('value returned is not a boolean! -{}-'.format(n))
		
def Integer(n):
	return int(n)
	
def String(n):
	return n 
	
def Float(n):
	return float(n)
	
def ListInteger(n):
	n_ = n.split(',')
	
	n__ = []
	for i in n_:
		j = [k for k in i if k not in '[ ]']
		n__.append(int(''.join(j)))
	
	return n__