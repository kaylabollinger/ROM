import numpy as np 

def local_linear_gradients(X,Y,n=None):
	"""Estimates gradients via local linear approximations.

	Note:
		This code follows that used in: https://github.com/paulcon/active_subspaces

	Args:
		X (ndarray): M-by-d 2D numpy array containing input training samples
		Y (ndarray): M length 1D numpy array containing the 1D output training samples
		n (int): number of nearest neighbors to use for local linear approximation

	Returns:
		DY (ndarray): M-by-d 2D numpy array containing local linear approximation of gradients for each training sample

	"""
	M, d = X.shape

	if M<=d:
		raise Exception('Not enough samples for local linear model: need M>d.')

	if n is None:
		n = int(min(M,2.*d))
	else:
		if not isinstance(n,int):
			raise TypeError('n must be an integer.')
		if n<=d or n>M:
			raise Exception('n must be in [d+1,M].')

	DY = []
	for x in X:
		dist = np.sum((X-x)**2,axis=1) # calculate distance from current point
		ind = np.argsort(dist) # determine nearest neighbors
		ind = ind[dist != 0] # ignore current point
		A = np.concatenate([np.ones((n,1)),X[ind[:n],:]],axis=1) # ones for affine approximation
		b = Y[ind[:n]]
		sol = np.linalg.lstsq(A,b)[0]
		DY.append(sol[1:])
	DY = np.stack(DY,axis=0)

	return DY
