import numpy as np
from scipy.spatial.transform import Rotation as R


def S(v):
	return np.array([[ 0.00, -v[2],  v[1]],
				     [ v[2],  0.00, -v[0]],
					 [-v[1],  v[0],  0.00]])


class Camera:
	def __init__(self, camera_params):
		# Camera parameters
		self.img_width = camera_params['img_width']
		self.img_height = camera_params['img_height']
		self.ratio = self.img_width / self.img_height
		self.fov = camera_params['fov']
		self.fps = camera_params['fps']

		## Camera states parameters
		# Position and orientation
		self.tf = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 1.57])

		# Rotation matrix
		self.Rrc = R.from_euler('xyz', self.tf[3:]).as_matrix()
		self.RrcT = self.Rrc.T

		# Motion matrix
		self.Vrc = np.zeros((6,6))
		self.Vrc[:3, :3] = self.Rrc
		self.Vrc[3:, 3:] = self.Rrc
		self.Vrc[:3, 3:] = S(self.tf[:3]) @ self.Rrc

		theta = np.deg2rad(self.fov)  # vertical FOV in radians
		self.tan_half_fov = np.tan(theta / 2)

		self.fy = (self.img_height / 2) / np.tan(theta / 2)  # focal length in pixels (y)
		self.fx = self.fy * (self.img_width / self.img_height)   # focal length in pixels (x)
