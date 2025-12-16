import cv2
import re

import numpy as np
from numpy.linalg import inv, pinv, norm
from scipy.spatial.transform import Rotation as R

from dynamics import Dynamics

def S(v):
	return np.array([[ 0.00, -v[2],  v[1]],
					[ v[2],  0.00, -v[0]],
					[-v[1],  v[0],  0.00]])

class Robot:
	def __init__(self, dt):
		
		# No dynamics for the robot now !
		# self.dynamics = Dynamics(robot_params)

		### Robot states variables	
		# Position and orientation
		self.eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
		# Velocity and angular velocity
		self.nu = np.zeros(6)
		# Extended rotation matrix
		self.J = np.zeros((6, 6))

		### Camera states parameters
		# Position and orientation
		self.cam_eta = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 1.57])
		# Rotation matrix
		self.Rrc = R.from_euler('xyz', self.cam_eta[3:]).as_matrix()
		# Motion matrix
		self.Vrc = np.zeros((6,6))
		self.Vrc[:3, :3] = self.Rrc
		self.Vrc[3:, 3:] = self.Rrc
		self.Vrc[:3, 3:] = S(self.cam_eta[:3]) @ self.Rrc
		# Camera parameters
		self.cam_width = 320
		self.cam_height = 240
		self.cam_fov = 60
		self.cam_fps = 10		
		# theta = np.deg2rad(self.cam_fov)  # vertical FOV in radians
		# self.fy = (self.cam_height / 2) / np.tan(theta / 2)  # focal length in pixels (y)
		# self.fx = self.fy * (self.cam_width / self.cam_height)   # focal length in pixels (x)

		### IBVS parameters
		self.lambda_gain = 0.4

		### IBVS scene with 4 points
		markers_init_pos = np.array([
								[0.0, 0.5, -0.5],
								[0.0, 0.5,  0.5],
								[0.0, -0.5, 0.5],
								[0.0, -0.5, -0.5],
								], dtype=float)
		t_markers = [3.0, 0.0, 0.0]
		R_markers = R.from_euler('xyz', [0, 0.5, 0.5]).as_matrix()
		self.markers = np.array([t_markers + R_markers @ m for m in markers_init_pos])
		self.markers_color = np.array([
								[1.0, 0.0, 0.0],
								[1.0, 0.0, 0.0],
								[1.0, 0.0, 0.0],
								[1.0, 0.0, 0.0],
								], dtype=float)
		self.wanted_marker_pos = np.array([
								[0.3, -0.3],
								[0.3, 0.3],
								[-0.3, 0.3],
								[-0.3, -0.3],
								], dtype=float).flatten()

		### Simulation parameters
		self.dt = dt
		self.time = 0
		self.save_images = False
		# self.save_images = True

	# Useless function for now (No dynamics)
	def update(self, dt, env):
		self.dynamics.step(dt, env)

		self.eta = self.dynamic.eta
		self.nu = self.dynamic.eta

	def compute_J1(self):
		"""
		Rotation matrix from body frame to inertial (NED) frame.
		"""
		eta2 = self.eta[3:]
		sphi = np.sin(eta2[0])
		cphi = np.cos(eta2[0])
		stheta = np.sin(eta2[1])
		ctheta = np.cos(eta2[1])
		spsi = np.sin(eta2[2])
		cpsi = np.cos(eta2[2])

		return np.array([
			[ctheta * cpsi,
			sphi * stheta * cpsi - cphi * spsi,
			cphi * stheta * cpsi + sphi * spsi],

			[ctheta * spsi,
			sphi * stheta * spsi + cphi * cpsi,
			cphi * stheta * spsi - sphi * cpsi],

			[-stheta,
			sphi * ctheta,
			cphi * ctheta]
		])

	def compute_J2(self):
		"""
		J2 matrix from Fossen formalism (Euler angle kinematics).
		"""
		eta2 = self.eta[3:]

		sphi = np.sin(eta2[0])
		cphi = np.cos(eta2[0])
		ttheta = np.tan(eta2[1])
		ctheta = np.cos(eta2[1])

		return np.array([
			[1.0,      sphi * ttheta,      cphi * ttheta],
			[0.0,      cphi,               -sphi],
			[0.0,      sphi / ctheta,      cphi / ctheta]
		])

	def compute_J(self):
		self.J[0:3, 0:3] = self.compute_J1()
		self.J[3:6, 3:6] = self.compute_J2()

	def update_visual_servoing(self, img):
		"""
		Function computing IBVS given the rendered image from viewer
		
		:param img: Numpy array image
		"""
		self.visual_servo_multi_points(img)

	def detect_targets_in_image(self, img):
		"""
		Compute basic computer vision to extract the markers from the image and return there coordinates in the image plane
		
		:param img: Numpy array image
		"""
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		# Optional: save hsv images
		if self.save_images:
			safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', f"{self.time:.3f}")
			cv2.imwrite(f'z_images/hsv_{safe_name}.png', hsv)


		# Define color thresholds (HSV)
		color_ranges = {
			'red': [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
			'green': [([36, 50, 70], [89, 255, 255])],
			'blue': [([90, 50, 70], [128, 255, 255])]
		}

		h, w = hsv.shape[:2]
		points = []

		i = 0

		for color, ranges in color_ranges.items():
			mask = np.zeros((h, w), dtype=np.uint8)
			for lower, upper in ranges:
				lower = np.array(lower, dtype=np.uint8)
				upper = np.array(upper, dtype=np.uint8)
				mask += cv2.inRange(hsv, lower, upper)
			
			# Optional: save mask images
			if self.save_images:
				safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', f"{self.time:.3f}_{color}")
				cv2.imwrite(f'z_images/mask_{color}_{safe_name}.png', mask)

			# Find contours
			contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			if len(contours) == 0:
				continue

			# Take the largest contour (assume the dot)
			cnt = max(contours, key=cv2.contourArea)
			M = cv2.moments(cnt)
			if M["m00"] > 0:
				cx = int(M["m10"] / M["m00"])
				cy = int(M["m01"] / M["m00"])
				nx = cx
				ny = cy
				z = (self.Rrc.T @ (self.markers[i] - self.eta[:3]))[2]
				points.append([nx, ny, z])
			

		return np.array(points)  # order: [red, green, blue]

	def detect_target_exact(self) -> np.ndarray:
		"""
		Compute exact X, Y and Z coordinates of markers in camera reference frame even when outside of the camera field of view
		"""
		points = [self.Rrc.T @ self.Rwr.T @ (self.markers[i] - self.eta[:3]) for i in range(len(self.markers))]
		return np.array(points)

	def visual_servo_multi_points(self, img):
		"""
		Compute IBVS next camera speed
		
		:param img: Description
		"""

		self.time += self.dt
		self.Rwr = R.from_euler('xyz', self.eta[3:]).as_matrix()

		if self.save_images:
			safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', f"{self.time:.3f}")
			cv2.imwrite(f'z_images/image_{safe_name}.png', img)
			print("image_saved")
		
		points = self.detect_target_exact()
		xs = points[:, 0]/points[:, 2]
		ys = points[:, 1]/points[:, 2]
		zs = points[:, 2]

		print("x, y:")
		print(xs)
		print(ys)

		x_v = np.array([[x, y] for x, y in zip(xs, ys)]).flatten()
		e = self.wanted_marker_pos - x_v

		print("e:")
		print(e, norm(e))

		L_e = np.array([
				[-1/z,  	0,    x/z,  x * y,     -(1 + x**2),  y] +
				[ 0, 	   -1/z,  y/z,  1 + y**2,  -x * y,      -x]
			for x, y, z in zip(xs, ys, zs)
		])

		L_e = L_e.reshape((len(points)*2, 6))

		if L_e.shape[0] == L_e.shape[1]:
			L_e_p = inv(L_e)	# Matrix inverse if the matrix is square
		else:
			L_e_p = pinv(L_e)	# Moore-Penrose inverse


		self.nu = self.lambda_gain * self.Vrc @ (L_e_p @ e)


		self.compute_J()

		self.eta += self.dt * self.J @ self.nu
		self.eta2 = self.eta[3:]
