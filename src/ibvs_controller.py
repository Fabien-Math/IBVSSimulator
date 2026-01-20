import numpy as np
from numpy.linalg import inv, pinv, norm
import re
import cv2
from scipy.spatial.transform import Rotation as R


class IBVSController:
	def __init__(self, mission_params, camera):

		self.camera = camera
		self.img = None
		self.new_img = False
		self.xs = None
		self.ys = None

		self.world = None

		### IBVS parameters
		self.lambda_gain = mission_params['lambda']
		self.tolerance = mission_params['tolerance']
		self.ratio_zs = mission_params['ratio_zs']
		self.img_computation = mission_params['img_computation']
		if self.img_computation:
			self.detect_target = self.detect_target_cv
		else:
			self.detect_target = self.detect_target_exact

		### IBVS scene
		self.wanted_marker_pos = mission_params['marker_pos_des'].flatten()

		### Simulation parameters
		self.time = 0
		self.save_images = mission_params['save_images']

		self.cmd = np.zeros(6)
		self.error_norm = 0

	def init_world(self, world):
		self.world = world
		self.markers = world.markers
		self.markers_color = world.marker_colors

	def update_img(self, img):
		self.img = img.copy()
		self.new_img = True

	def update(self, eta, time):
		"""
		Function computing IBVS given the rendered image from viewer

		:param eta: Robot state
		"""
		self.time = time
		if self.save_images and self.new_img:
			self.save_image()
			self.new_img = False

		points = self.detect_target(eta)
		zs = points[:, 2]
		self.xs = points[:, 0]
		self.ys = points[:, 1]

		x_v = np.array([[x, y] for x, y in zip(self.xs, self.ys)]).flatten()
		self.errors = self.wanted_marker_pos - x_v
		self.error_norm = norm(self.errors)

		L_e_z = np.array([
				[-1/z,  	0,    x/z,  x * y,     -(1 + x**2),  y] +
				[ 0, 	   -1/z,  y/z,  1 + y**2,  -x * y,      -x]
			for x, y, z in zip(self.xs, self.ys, zs)
		])

		z_s = 1
		L_e_s = np.array([
				[-1/z_s,  	0,    x/z_s,  x * y,     -(1 + x**2),  y] +
				[ 0, 	   -1/z_s,  y/z_s,  1 + y**2,  -x * y,      -x]
			for x, y in zip(self.xs, self.ys)
		])

		L_e_z = L_e_z.reshape((len(points)*2, 6))
		L_e_s = L_e_s.reshape((len(points)*2, 6))
		L_e = self.ratio_zs * L_e_s + (1 - self.ratio_zs) * L_e_z

		if L_e.shape[0] == L_e.shape[1]:
			if np.linalg.det(L_e):	# Another level of security
				L_e_p = inv(L_e)	# Matrix inverse if the matrix is square
			else:
				L_e_p = pinv(L_e)	# Moore-Penrose inverse
		else:
			L_e_p = pinv(L_e)	# Moore-Penrose inverse

 		# Keep track of the target in the center of the image
		x_cog, y_cog, z_cog = 0.0, 0.0, 0.0
		for xs, ys, zs_i in zip(self.xs, self.ys, zs):
			x_cog += xs
			y_cog += ys
			z_cog += zs_i
		x_cog /= len(self.xs)
		y_cog /= len(self.ys)
		z_cog /= len(zs)
		ratio = min(1, np.sqrt(x_cog**2 + y_cog**2))
		ratio = ratio * (ratio > 0.2)

		z_s = max(1, z_cog)
		L_center_p = np.linalg.pinv([
				[-1/z_s,  	0,    x_cog/z_s,  x_cog * y_cog,     -(1 + x_cog**2),  y_cog],
				[ 0, 	   -1/z_s,  y_cog/z_s,  1 + y_cog**2,  -x_cog * y_cog,      -x_cog]])

		cog_error = - np.array([x_cog, y_cog])

		ibvs_cmd = self.lambda_gain * self.camera.Vrc @ (L_e_p @ self.errors)
		centering_cmd = self.lambda_gain * self.camera.Vrc @ (L_center_p @ cog_error)

		self.cmd = (1 - ratio) * ibvs_cmd + ratio * centering_cmd



	def detect_target_exact(self, eta) -> np.ndarray:
		"""
		Compute exact X, Y and Z coordinates of markers in camera reference frame even when outside of the camera field of view
		"""
		Rwr = R.from_euler('xyz', eta[3:]).as_matrix()
		points = np.array([self.camera.RrcT @ Rwr.T @ (self.markers[i] - eta[:3]) for i in range(len(self.markers))])
		zs = points[:, 2]

		points[:, 1] = (points[:, 1] / zs) / self.camera.tan_half_fov
		points[:, 0] = (points[:, 0] / zs) / (self.camera.tan_half_fov * self.camera.ratio)

		return np.array(points)

	def detect_target_cv(self, eta):
		# Convert to HSV
		output = self.img.copy()

		# Blur to reduce noise (important at low resolution)
		blurred = cv2.GaussianBlur(self.img, (5, 5), 0)

		# Convert to HSV
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

		# Red color ranges (HSV)
		lower_red1 = np.array([0, 120, 70])
		upper_red1 = np.array([10, 255, 255])

		lower_red2 = np.array([170, 120, 70])
		upper_red2 = np.array([180, 255, 255])

		# Create masks
		mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
		mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
		mask = mask1 | mask2

		# Morphology tuned for 320x240
		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

		# Find contours
		contours, _ = cv2.findContours(
			mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)

		points = []

		for cnt in contours:
			(cx, cy), r = cv2.minEnclosingCircle(cnt)

			cx = cx / (self.camera.img_width / 2) - 1
			cy = cy / (self.camera.img_height / 2) - 1

			points.append((cx, cy, 1/r))
			# Draw result
			cv2.circle(output, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
			cv2.drawContours(output, [cnt], -1, (255, 0, 0), 1)

		if self.save_images:
			self.save_image(output, name='output')

		return np.array(points)


	def save_image(self, img=None, name='image'):
		safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', f"{self.time:.3f}")
		if img is None:
			cv2.imwrite(f'z_images/{name}_{safe_name}.png', self.img)
		else:
			cv2.imwrite(f'z_images/{name}_{safe_name}.png', img)
		print("image_saved:", f'z_images/{name}_{safe_name}.png')
