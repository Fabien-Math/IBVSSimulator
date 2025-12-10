import cv2
import re

import numpy as np
from scipy.spatial.transform import Rotation as R

from dynamics import Dynamics

class Robot:
	def __init__(self, robot_params, dt):
		# self.dynamics = Dynamics(robot_params)		
		# Position and orientation
		self.eta = np.array([0.0, -2.0, 0.0, 0.0, 0.0, 1.57]) 
	
		# Velocity and angular velocity
		self.nu = np.zeros(6)

		self.lambda_gain = 0.8
		# self.markers = np.array([
		# 						[1.0, 0.7, 0.0, 0.0, 0.0, 0.0],
		# 						[1.5, 0.7, 0.0, 0.0, 0.0, 0.0],
		# 						[1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
		# 						[1.5, 0.5, 0.5, 0.0, 0.0, 0.0],
		# 						], dtype=float)
		# self.wanted_marker_pos = np.array([
		# 						[0.5, 0.5],
		# 						[-0.5, 0.5],
		# 						[-0.5, -0.5],
		# 						[0.5, -0.5],
		# 						], dtype=float)

		self.markers = np.array([
								[1.0, 0.7, 0.0, 0.0, 0.0, 0.0],
								[0.5, 0.7, 0.0, 0.0, 0.0, 0.0],
								[1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
								], dtype=float)
		self.markers_color = np.array([
								[1.0, 0.0, 0.0],
								[0.0, 1.0, 0.0],
								[0.0, 0.0, 1.0],
								], dtype=float)
		self.wanted_marker_pos = np.array([
								[100, 50],
								[200, 150],
								[50, 150],
								], dtype=float)

		self.dt = dt
		self.time = 0


		self.cam_width = 320
		self.cam_height = 240
		self.cam_fov = 60

		
		theta = np.deg2rad(self.cam_fov)  # vertical FOV in radians
		self.fy = (self.cam_height / 2) / np.tan(theta / 2)  # focal length in pixels (y)
		self.fx = self.fy * (self.cam_width / self.cam_height)   # focal length in pixels (x)

		# self.save_images = False
		self.save_images = False


	def update(self, dt, env):
		
		self.dynamics.step(dt, env)

		self.eta = self.dynamic.eta
		self.nu = self.dynamic.eta

	def update_visual_servoing(self, img):
		self.visual_servo_multi_points(img)
		
	# ------------------------------------------------------------
	# Target detection
	# ------------------------------------------------------------
	def detect_targets_in_image(self, img):
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# Define color thresholds (HSV)
		color_ranges = {
			'red': [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
			'green': [([36, 50, 70], [89, 255, 255])],
			'blue': [([90, 50, 70], [128, 255, 255])]
		}

		h, w = hsv.shape[:2]
		points = []

		for color, ranges in color_ranges.items():
			mask = np.zeros((h, w), dtype=np.uint8)
			for lower, upper in ranges:
				lower = np.array(lower, dtype=np.uint8)
				upper = np.array(upper, dtype=np.uint8)
				mask += cv2.inRange(hsv, lower, upper)
			
			# Optional: save mask images
			if self.save_images:
				safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', f"{self.time:.3f}_{color}")
				cv2.imwrite(f'VisualServoing/z_images/mask_{color}_{safe_name}.png', mask)

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
				nx = (cx - w / 2) / self.fx
				ny = (cy - h / 2) / self.fy
				points.append([nx, ny])

		return np.array(points)  # order: [red, green, blue]



	# ------------------------------------------------------------
	# Visual servoing (IBVS)
	# ------------------------------------------------------------
	def visual_servo_multi_points(self, img):

		self.time += self.dt
		
		points = self.detect_targets_in_image(img)
		# print(len(points), points.flatten())
		if len(points) < 3:
			# print(f"Only {len(points)} points found!")
			return
		
		if self.save_images:
			safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', f"{self.time:.3f}")
			cv2.imwrite(f'VisualServoing/z_images/image_{safe_name}.png', img)
			print("image_saved")
		
		
		wanted_points = self.wanted_marker_pos

		e = points.flatten() - wanted_points.flatten()

		
		z_star = 1
		L_e_star = np.array([
				[-1/z_star,  0,  p[0]/z_star,  p[0] * p[1],  -(1 + p[0]**2),  p[1]] +
				[ 0, -1/z_star,  p[1]/z_star,   1 + p[1]**2,  -p[0] * p[1],  -p[0]]
			for p in points
		])


		# Compute rotation matrix from camera orientation (self.eta[3:])
		rotation_matrix = R.from_euler('xyz', self.eta[3:]).as_matrix()

		zs = []

		for i in range(3):
			# Vector from camera to marker in world frame
			front_vec = rotation_matrix @ np.array([1.0, 0, 0])
			vec = self.markers[i][:3] - self.eta[:3]
			
			# Express vector in camera frame
			vec_cam = rotation_matrix.T @ vec  # transpose = inverse rotation
			# Depth along camera Z-axis
			zs.append(vec_cam[0])

		zs = np.array(zs)  # final 1D array of depths
		L_e_e = np.array([
				[-1/z,  0,  p[0]/z,  p[0] * p[1],  -(1 + p[0]**2),  p[1]] +
				[ 0, -1/z,  p[1]/z,   1 + p[1]**2,  -p[0] * p[1],  -p[0]]
			for z, p in zip(zs, points)
		])

		L_e = 0.5 * (L_e_e + L_e_star).reshape((len(points)*2, 6))

		if L_e.shape[0] == L_e.shape[1]:
			L_e_p = np.linalg.inv(L_e)
		else:
			L_e_p = np.linalg.inv(L_e.T @ L_e) @ L_e.T

		self.nu = - self.lambda_gain * L_e_p @ e

		# print("Error")
		print(e, flush=True)
		# print(points)
		# print(wanted_points)
		print(zs, flush=True)
		# print("nu:", self.nu)

		self.eta += self.dt * self.nu