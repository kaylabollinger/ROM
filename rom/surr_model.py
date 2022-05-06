import numpy as np 
from sklearn import linear_model
from .utils import check_2D, check_1D, print_epoch, rel_error

import torch
import torch.nn as nn

from pymanopt.manifolds import Grassmann
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

class RF():
	"""Random Feature Expansion surrogate model.

	Note: 
		All attributes are set after the "train" method is called.

	Attributes:
		c (ndarray): N length 1D numpy array containing coefficients of learned RF model
		Omega (ndarray): N-by-d 2D numpy array containing weights of the RF model
		bias (ndarray): N length 1D numpy array containing biases of the RF model
		scale_A_normalize (ndarray): N length 1D numpy array to normalize columns of A_train (RF matrix with training data)

	"""

	def __init__(self):
		self.c = None
		self.Omega = None
		self.bias = None
		self.scale_A_normalize = None

	def train(self,dataset_train,N=None,s=None,alpha=0.001):
		"""Trains RF model by learning coefficients c via ridge regression.

		Note:
			Assumes output data is 1D.

		Args:
			X_train (ndarray): M-by-d 2D numpy array containing input training data
			Y_train (ndarray): M length 1D numpy array containing output training data
			N (int): number of weights to use for RF model
			alpha (float): regularizing hyperparameter for ridge regression model

		"""
		X_train = check_2D(dataset_train[0])
		Y_train = check_1D(dataset_train[1])

		M, k = X_train.shape

		if N is None:
			# if N is not specified, use default regimes
			if M > 5000:
				# underparameterized regime if number of training samples is too large
				N = int(np.floor(M/10))
			else:
				# else work in overparameterized regime
				N = 5*M

		if s is None:
			s = int(np.ceil(k/2))
		else:
			if s > k:
				raise TypeError('sparsity value s must be <= input dimension')

		self.construct_weights(X_train,k,N,s)

		A_train = self.construct_A(X_train)

		clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, max_iter=5000, tol=0.0001)
		clf.fit(A_train, Y_train)
		self.c = clf.coef_

	def predict(self,X):
		"""Calculates output of trained RF model.
		
		Args:
			X (ndarray): ?-by-d 2D numpy array containing ? number of input data points

		Returns:
			Y (ndarray): ? length 1D numpy array containing predicted output values

		"""

		A = self.construct_A(X)
		Y = np.matmul(A,self.c.T)
		return Y

	def construct_weights(self,X_train,k,N,s):
		"""Constructs weights and biases for RF model.

		Args:
			X_train (ndarray): M-by-d 2D numpy array containing input training data
			k (int): dimension of input data
			N (int): number of weights to use for RF model
			s (int): sparsity for weights Omega

		"""
		Omega_keep = [] 
		bias_keep = []
		scale_A_normalize_keep = []
		N_check = 10*N
		ct_check = 0
		ct_pass = 0
		while ct_check < N_check and ct_pass < N:
			ct_check += 1

			Omega = np.random.uniform(low=-1,high=1,size=(1,k))

			ind_zero = np.random.choice(k,k-s,replace=False)
			Omega[:,ind_zero] = 0.

			bias = np.random.uniform(low=-1,high=1)

			A_temp = self.phi(np.matmul(X_train,Omega.T)+bias)
			mag = np.linalg.norm(A_temp) 
			if mag > 1e-5:
				ct_pass += 1
				Omega_keep.append(Omega)
				bias_keep.append(bias)
				scale_A_normalize_keep.append(mag)

		if ct_check == N_check:
			print('Warning: not enough suitable weights.')

		self.Omega = np.concatenate(Omega_keep,axis=0)
		self.bias = np.array(bias_keep)
		self.scale_A_normalize = np.array(scale_A_normalize_keep)

	def construct_A(self,X):
		"""Constructs random feature matrix A.

		Args:
			X (ndarray): ?-by-d 2D numpy array containing ? number of input data points

		Returns:
			A (ndarray): ?-by-N 2D numpy array representing the random feature matrix A

		"""
		A = self.phi(np.matmul(X,self.Omega.T)+self.bias)/self.scale_A_normalize
		return A

	def phi(self,nodes):
		"""
		Activation function phi for RF model.

		Args:
			nodes (ndarray): N length 1D numpy array containing nodal values

		Returns:
			out (ndarray): N length 1D numpy array containing nodes passed through activation function phi

		"""
		out = np.maximum(nodes,0.)
		return out


class NN_alt():
	"""Shallow ReLU Network with alternating minimization surrogate model.

	Attributes:
		net (): two layer ReLU network
		optimizer (): optimizer for network
		loss_func (): loss function for network
		lr (float): learning rate
		lr_decay (float): learning rate decay
		alpha (float): regularization parameter (ell 2 regularizer)
		U (ndarray): d-by-k 2D numpy array to reduce input dimension
		loss_train (list): contains training loss values at each epoch
		loss_val (list): contains validation loss values at each epoch

	"""
	def __init__(self,U,dim_layers,lr=0.001,lr_decay=0.9,alpha=0.001):
		if len(dim_layers)!=3:
			raise TypeError('Exactly 3 dimension values must be given: input, hidden, and output dimension sizes.')
		self.net = self._ShallowNet(input_dim=dim_layers[0],
									hidden_dim=dim_layers[1], 
									output_dim=dim_layers[2])
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr,weight_decay=alpha) 
		self.loss_func = nn.MSELoss(reduction='mean')
		self.lr = lr 
		self.lr_decay = lr_decay
		self.alpha = alpha
		self.U = U
		self.loss_train	= []
		self.loss_val = []

	def train(self,dataset_train,dataset_val=None,num_outer=10,
			  num_epoch=5000,batch_size=16,record=True,verbose=2):
		"""Trains NN model via alternating minimization scheme.

		Args:
			dataset_train (tuple,ndarray): [X_train,Y_train], numpy array containing input/output training data
			dataset_val (tuple,ndarray): [X_val,Y_val], numpy array containing input/output validation data
			num_outer (int): number of outer iterations to perform
			num_epoch (int): number of inner iterations for NN training to perform
			batch_size (int): number of training samples in each training batch
			record (bool): choose to record loss during training or not
			verbose (int): amount of information to print; 0 = nothing printed; 1 = loss values printed every 1000 epochs; 2 = loss values printed every 100 epochs; >2 = loss values printed every epoch

		"""
		X_train = check_2D(dataset_train[0])
		Y_train = check_2D(dataset_train[1])

		print('Training parameters:')
		print(f'\tNumber of outer iterations = {num_outer}')
		print(f'\tNumber of inner iterations for NN = {num_epoch}')
		print(f'\tBatch size for training NN = {batch_size}')

		# begin alternating minimization process
		for ite in range(num_outer-1):
			print(f'\nOuter iteration: {ite+1}\n')

			# train NN
			self.train_NN(dataset_train,dataset_val,num_epoch,batch_size,record,verbose)  

			# update optimizer
			self.optimizer = torch.optim.Adam(
								self.net.parameters(), 
								lr=(self.lr_decay**(ite))*self.lr,
								weight_decay=self.alpha
								) 

			# train subspace
			self.train_sub(X_train,Y_train)
		
		print(f'\nOuter iteration: {num_outer}\n')
		self.train_NN(dataset_train,dataset_val,num_epoch,batch_size,record,verbose)  

	def predict(self,X):
		"""Calculates output of trained NN model.

		Args:
			X (ndarray): ?-by-d 2D numpy array containing input data

		Returns:
			Y (ndarray): ?-by-d2 2D numpy array containing output data

		"""
		X_tensor = torch.from_numpy(X).float()
		U_tensor = torch.from_numpy(self.U).float()

		with torch.no_grad():
			Y_tensor = self.net(X_tensor,U_tensor)

		Y = Y_tensor.numpy()
		return Y

	def train_NN(self,dataset_train,dataset_val,num_epoch,batch_size,record,verbose):
		"""Trains and validates NN.

		Args:
			dataset_train (tuple,ndarray): [X_train,Y_train], numpy array containing input/output training data
			dataset_val (tuple,ndarray): [X_val,Y_val], numpy array containing input/output validation data
			num_outer (int): number of outer iterations to perform
			num_epoch (int): number of inner iterations for NN training to perform
			batch_size (int): number of training samples in each training batch
			record (bool): choose to record loss during training or not
			verbose (int): amount of information to print; 0 = nothing printed; 1 = loss values printed every 1000 epochs; 2 = loss values printed every 100 epochs; >2 = loss values printed every epoch

		Returns:
			loss_train (list): contains training loss at each epoch
			loss_val (list): contains validation loss at each epoch

		"""

		# convert data to torch tensor
		X_train_tensor = torch.from_numpy(dataset_train[0]).float()
		Y_train_tensor = torch.from_numpy(dataset_train[1]).float()
		if dataset_val is not None:
			X_val_tensor = torch.from_numpy(dataset_val[0]).float()
			Y_val_tensor = torch.from_numpy(dataset_val[1]).float()
		U_tensor = torch.from_numpy(self.U).float()

		# create batches for training data
		train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
		train_loader = torch.utils.data.DataLoader(
			dataset=train_dataset, batch_size=batch_size, shuffle=True)

		if record:
			with torch.no_grad():
				self.loss_train.append(rel_error(Y_train_tensor.numpy(),self.net(X_train_tensor,U_tensor).numpy()))
				if dataset_val is not None:
					self.loss_val.append(rel_error(Y_val_tensor.numpy(),self.net(X_val_tensor,U_tensor).numpy()))

			if dataset_val is not None:
				print_epoch(verbose,0,num_epoch,self.loss_train[-1],self.loss_val[-1])
			else:
				print_epoch(verbose,0,num_epoch,self.loss_train[-1],None)

		# run training
		for epoch in range(num_epoch):
			for (X_train_batch, Y_train_batch) in train_loader:  
				def closure():
					self.optimizer.zero_grad()
					loss = self.loss_func(self.net(X_train_batch,U_tensor), Y_train_batch)
					loss.backward()
					return loss
				self.optimizer.step(closure)

			if record:
				with torch.no_grad():
					self.loss_train.append(rel_error(Y_train_tensor.numpy(),self.net(X_train_tensor,U_tensor).numpy()))
					if dataset_val is not None:
						self.loss_val.append(rel_error(Y_val_tensor.numpy(),self.net(X_val_tensor,U_tensor).numpy()))

				if dataset_val is not None:
					print_epoch(verbose,epoch+1,num_epoch,self.loss_train[-1],self.loss_val[-1])
				else:
					print_epoch(verbose,epoch+1,num_epoch,self.loss_train[-1],None)

	def train_sub(self,X_train,Y_train):
		"""Trains subspace U. 

		Args: 
			X_train (ndarray): M-by-d 2D numpy array containing input training data
			Y_train (ndarray): M-by-d2 2D numpy array containing output training data

		"""
		m, n = self.U.shape

		# instantiate the Grassmann manifold  
		manifold = Grassmann(m, n)

		# define cost and gradient functions
		cost = lambda U: self.sub_res(U, X_train, Y_train)
		grad = lambda U: self.sub_dres(U, X_train, Y_train)

		# instantiate optimization problem over Grassman manifold
		problem = Problem(manifold=manifold, cost=cost, egrad=grad, verbosity=0)
		solver = SteepestDescent(logverbosity=0)

		self.U = solver.solve(problem, x=self.U) # update U

	def sub_res(self,U,X_train,Y_train):
		"""Defines cost function w.r.t. U. 

		Note:
			This function is defined for the pymanopt optimization problem. Must take in U as variable.

		Args: 
			U (ndarray): d-by-k 2D numpy array containing reduced basis
			X_train (ndarray): M-by-d 2D numpy array containing input training data
			Y_train (ndarray): M-by-d2 2D numpy array containing output training data

		Returns:
			out (float): cost value for given U

		"""

		# convert data to torch tensor
		X_train = torch.from_numpy(X_train).float()
		Y_train = torch.from_numpy(Y_train).float()

		U = torch.from_numpy(U).float()

		with torch.no_grad():
			out = self.loss_func(self.net(X_train,U),Y_train)

		return out

	def sub_dres(self,U,X_train,Y_train):
		"""Defines derivative of cost function w.r.t. U. 

		Note:
			This function is defined for the pymanopt optimization problem. Must take in U as variable.

		Args:  
			U (ndarray): d-by-k 2D numpy array containing reduced basis
			X_train (ndarray): M-by-d 2D numpy array containing input training data
			Y_train (ndarray): M-by-d2 2D numpy array containing output training data

		Returns:
			(ndarray): d-by-k 2D numpy array containing reduced basis

		"""

		# convert data to torch tensor
		X_train = torch.from_numpy(X_train).float()
		Y_train = torch.from_numpy(Y_train).float()

		U = torch.from_numpy(U).float()

		U.requires_grad = True

		loss = self.loss_func(self.net(X_train,U),Y_train)
		loss.backward()

		return U.grad.numpy()

	def rel_error_tensor(self,Y_true,Y_calc):
		"""Calculates relative error (via ell 2 norm) between 2 PyTorch tensors.

		Args:
			Y_true (ndarray): numpy array containing true values
			Y_calc (ndarray): numpy array containing calculated/predicted values

		Returns:
			(float): relative error between Y_true and Y_calc

		"""
		return np.sqrt(self.loss_func(Y_true,Y_calc)/self.loss_func(Y_true,torch.zeros(Y_true.size(),requires_grad=False)))


	class _ShallowNet(nn.Module):
		'''Shallow ReLU network structure.
		
		Attributes:
			h_0 (): first linear layer
			h_last (): second linear layer
			sigmoid (): ReLU activation function

		Args:
			input_dim (int): dimension of input data
			hidden_dim (int): dimension of hidden layer
			output_dim (int): dimension of output data

		'''
		def __init__(self, input_dim, hidden_dim, output_dim):
			super().__init__()
			self.h_0 = nn.Linear(input_dim, hidden_dim) 
			self.h_last = nn.Linear(hidden_dim, output_dim)
			self.sigmoid = nn.ReLU()
			return
		def forward(self, x0, U):
			"""

			Args:
				x0 (torch tensor): M-by-d 2D torch tensor containing input data
				U (torch tensor): d-by-k 2D torch tensor for dimension reduction step

			Returns:
				x3 (torch tensor): M-by-d2 2D torch tensor containing predicted output

			"""
			x0 = torch.matmul(x0, U)
			x1 = self.h_0(x0)
			x2 = self.sigmoid(x1)
			x3 = self.h_last(x2)
			return x3
