import numpy as np

def AS(DY,k):
	"""Constructs a reduced subspace via active subspaces.

	Args:
		DY (ndarray): M-by-d 2D numpy array containing gradients for each training sample
		k (int): reduced dimension size

	Returns:
		U (ndarray): d-by-k 2D numpy array containing reduced basis vectors

	"""

	U, _, _, = np.linalg.svd(DY.transpose(),full_matrices=False)
	U = U[:,:k]

	return U

def POD(X,k):
	"""Constructs a reduced subspace via active subspaces.

	Args:
		X (ndarray): M-by-d 2D numpy array containing gradients for each training sample
		k (int): reduced dimension size

	Returns:
		U (ndarray): d-by-k 2D numpy array containing reduced basis vectors

	"""

	U, _, _, = np.linalg.svd(X.transpose(),full_matrices=False)
	U = U[:,:k]

	return U
