import numpy as np 

def check_2D(X):
	"""
	Enforces requirement that X be a 2D numpy array.

	Parameters:
		- X : ndarray
			ND numpy array containing data points

	Returns:
		- X : ndarray
			unaltered 2D numpy array

	Notes:
	"""
	if len(X.shape)!=2:
		raise TypeError('Data must be stored in 2D numpy arrray.')
	return X	

def check_1D(X):
	"""
	Enforces requirement that X be a 1D numpy array.

	Parameters:
		- X : ndarray
			ND numpy array containing data points

	Returns:
		- X : ndarray
			unaltered 1D numpy array

	Notes:
	"""
	# if len(X.shape)!=1:
	# 	raise TypeError('Data must be stored in 1D numpy arrray.')
	return X	

def rel_error(Y_true,Y_calc):
	"""
	Calculates relative error (via ell 2 norm) between 2 arrays.

	Parameters:
		- Y_true : ndarray
			numpy array containing true values
		- Y_calc : ndarray
			numpy array containing calculated/predicted values

	Returns:
		- out : float
			relative error between Y_true and Y_calc

	Notes:
	"""
	out = np.linalg.norm(Y_true-Y_calc)/np.linalg.norm(Y_true)
	return out

def print_epoch(verbose,epoch,num_epoch,loss_train,loss_val=None,overwrite=False):
	'''
	Prints training/validation error at given epoch.

	Parameters:
		- 

	Returns:
		None

	Notes:
	'''
	line = f'Epoch {epoch}/{num_epoch} | Train Rel Loss {loss_train:8f}'
	if loss_val is not None:
		line += f' | Validation Rel Loss: {loss_val:.8f}'

	if overwrite:
		print_end = '\r'
	else:
		print_end = '\n'

	if verbose:
		if verbose>2:
			print(line,end=print_end)
		elif verbose>1:
			if epoch % 100 == 0:
				print(line,end=print_end)
		else:
			if epoch % 1000 == 0:
				print(line,end=print_end)
