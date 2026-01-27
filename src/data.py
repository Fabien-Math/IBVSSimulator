import numpy as np

class Data:
	def __init__(self):
		self.tfs = np.random.random((1000, 6))
		self.tf = np.zeros(6)

	def update(self):
		# update function called inside the OpenGL loop
		self.tfs = np.random.random((1000, 6))
