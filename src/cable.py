from cable_joint import CableJoint
import numpy as np
from numpy.linalg import norm

def normalize(v):
	# Need to see what to do when norm = 0
	return v / norm(v)

def c1(x, length, d):
	return x - length / (2 * np.arcsin(d / (2 * x)))

def c2(x, length, d):
	return x - length / (2 * np.pi - 2 * np.arcsin(d / (2 * x)))


def newton_raphson_circle(x0, length, d):
	n = 0
	n_max = 1e3
	err = 1e-3
	h = 1e-6
	x = x0

	if (length >= np.pi * d / 2):
		while (abs(c2(x, length, d)) > err):
			n += 1
			xp = (c2(x+h, length, d) - c2(x-h, length, d)) / (2 * h)
			x -= c2(x, length, d) / xp
			if (n > n_max):
				return None
	else:
		while (abs(c1(x, length, d)) > err):
			n += 1
			xp = (c1(x+h, length, d) - c1(x-h, length, d)) / (2 * h)
			x -= c1(x, length, d) / xp
			if (n > n_max):
				return None
	return x


class Cable:
	def __init__(self, anchor1_pos, anchor2_pos, length, radius, n_subdiv, linear_mass, bending_coef, jokobsen_params, anchors_fixed, color=(1.0, 0.0, 0.0)):
		self.length = length
		self.radius = radius
		self.linear_mass = linear_mass
		self.n = n_subdiv

		self.jakobsen_params = jokobsen_params
		self.bending_coef = bending_coef

		self.time = 0.0
		self.last_time = 0.0
		self.freq = 100.0

		self.anchor1 = anchor1_pos
		self.anchor2 = anchor2_pos

		self.anchors_fixed = anchors_fixed
		self.joints = [None] * self.n

		self.color = color
		# halfHeight = length/2)
		# graMesh = OpenGLContent::BuildCylinder((GLfloat)(radius), (GLfloat)(length), (unsigned int)btMax(ceil(2.0*np.pi*radius/0.1), 32.0)) #Max 0.1 m cylinder wall slice width
		# phyMesh[i] = OpenGLContent::BuildCableMesh((GLfloat)(_radius), pos, (unsigned int)_nSlice) #Max 0.1 m cylinder wall slice width

		self.init_cable()

		self.update_positions()


	def update_positions(self):
		self.positions = [joint.pos for joint in self.joints]

	def init_cable(self):
		seg_length = self.length / (self.n-1)
		seg_mass = self.linear_mass * seg_length

		self.joints[0] = CableJoint(0, -1, 1, None, self.radius, seg_length, seg_mass, self.bending_coef, self.anchors_fixed[0])
		for i in range(1, self.n-1):
			self.joints[i] = CableJoint(i, i-1, i+1, None, self.radius, seg_length, seg_mass, self.bending_coef, False)
		self.joints[self.n - 1] = CableJoint(self.n - 1, self.n - 2, -1, None, self.radius, seg_length, seg_mass, self.bending_coef, self.anchors_fixed[1])

		self.init_cable_joints_pos()


	def init_cable_joints_pos(self):
		a1 = self.anchor1
		a2 = self.anchor2

		d = np.linalg.norm(a2 - a1)

		# Straight cable
		if abs(self.length - d) < 0.05 * self.length:
			for i in range(self.n):
				t = i / (self.n - 1)
				p = (1 - t) * a1 + t * a2
				self.joints[i].pos = p.copy()
				self.joints[i].last_pos = p.copy()
			return

		# Radius from arc length
		R = newton_raphson_circle(d / 2 + 1e-3, self.length, d)
		if R is None:
			raise ValueError("ERROR ! Cable not initialized correctly!")

		# Circle center correction
		sign = 1
		if np.pi * R > self.length:
			sign = -1

		# Chord direction
		u = normalize(a2 - a1)

		# Downward reference
		ref = np.array([0.0, 0.0, 1.0])
		n = normalize(np.cross(u, ref))
		v = np.cross(n, u)

		# Circle center
		h = np.sqrt(R * R - (d / 2) ** 2)
		mp = 0.5 * (a1 + a2)
		C = mp + sign * h * v

		# Endpoint directions
		e1 = normalize(a1 - C)
		e2 = normalize(a2 - C)

		angle = np.arccos(np.clip(np.dot(e1, e2), -1.0, 1.0))
		minor_len = R * abs(angle)
		major_len = R * (2 * np.pi - abs(angle))

		if abs(self.length - major_len) < abs(self.length - minor_len):
			angle = angle - np.sign(angle) * 2 * np.pi

		# Place joints
		for i in range(self.n):
			t = i / (self.n - 1)
			phi = t * angle
			p = C + R * (np.cos(phi) * e1 + sign * np.sin(phi) * np.cross(n, e1))
			self.joints[i].pos = p.copy()
			self.joints[i].last_pos = p.copy()


	def get_cable_length(self):
		totalLength = 0.0
		for i in range(self.n-1):
			d = norm(self.joints[i+1].pos - self.joints[i].pos)
			totalLength += d

		return totalLength


	def update(self, dt, world):
		self.time += dt
		self.get_cable_length()
		for i in range(self.n):
			self.joints[i].compute_forces(self.joints, world)
		for i in range(self.n):
			self.joints[i].update_pos(dt)
		self.jakobsen()

	def jakobsen(self):
		for _ in range(self.jakobsen_params['n_iter']):
			for j in range(self.n - 1):
				p1 = self.joints[j].pos
				p2 = self.joints[j+1].pos

				delta = p2 - p1
				dist = norm(delta)
				if (dist == 0):
					continue

				diff = dist - self.joints[j].length
				corr = normalize(delta) * 0.5 * diff * self.jakobsen_params['penalty_coef']

				if not self.joints[j].fixed:
					self.joints[j].pos += corr
				if not self.joints[j+1].fixed:
					self.joints[j+1].pos -= corr
