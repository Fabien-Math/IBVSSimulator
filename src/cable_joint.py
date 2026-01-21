import numpy as np

def normalize(v):
	# Need to see what to do when norm = 0
	return v / np.linalg.norm(v)

def angle(v1, v2):
	v1_u = normalize(v1)
	v2_u = normalize(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class CableJoint:
	def __init__(self, id, j1, j2, position,
					radius, length, mass, bending_coef, fixed):
		self.id = id
		self.j1 = j1
		self.j2 = j2

		self.pos = position
		self.last_pos = position
		self.vel = np.zeros(3)
		self.acc = np.zeros(3)
		self.force = np.zeros(3)

		self.radius = radius
		self.length = length
		self.mass = mass

		self.volume = self.length * np.pi * self.radius * self.radius
		self.surface = 2.0 * self.length * self.radius

		self.kt = bending_coef
		self.cd_cylinder = 1.17

		self.fixed = fixed

	def compute_forces(self, joints, world):
		if (self.fixed):
			self.force = np.zeros(3)
			return

		# Volumetric forces
		f_weigth = self.mass * world.gravity
		f_archi = - self.volume * world.ocean.water_density * world.gravity

		self.force = f_weigth + f_archi

		# Bending force
		if (self.j1 >= 0 and self.j2 >= 0):
			v1 = normalize(joints[self.j1].pos - self.pos)
			v2 = normalize(joints[self.j2].pos - self.pos)

			alpha = angle(v1, v2) - np.pi
			moment_axis = np.cross(v1, v2)
			if (np.linalg.norm(moment_axis) > 1e-6):
				normalize(moment_axis)
				moment = 2 * self.kt * (np.tan(alpha / 4) / np.cos(alpha / 4))
				f_bending = moment * 0.5 * self.length * np.cross(moment_axis, v2)
				self.force += f_bending

		# Fluid friction
		rel_vel = world.ocean.fluid_vel[:3] - self.vel
		u = np.zeros(3)

		if (self.j1 >= 0 and self.j2 >= 0):
			u = 0.5 * (normalize(self.pos - joints[self.j1].pos) + normalize(joints[self.j2].pos - self.pos))
		elif (self.j1 >= 0 and self.j2 < 0):
			u = normalize(self.pos - joints[self.j1].pos)
		elif (self.j1 < 0 and self.j2 >= 0):
			u = normalize(joints[self.j2].pos - self.pos)

		if (np.linalg.norm(u)):
			rel_vel_sign = np.sum(rel_vel)
			normal = rel_vel_sign * np.cross(u, np.cross(rel_vel, u))
			f_fluid = 0.5 * world.ocean.water_density * self.surface * self.cd_cylinder * np.abs(rel_vel) * rel_vel
			self.force += f_fluid

		damping_coef = min(abs(self.mass) / 10.0, 0.01)
		damping_force = - damping_coef * self.vel
		self.force += damping_force

		# Cable traction
		# if (self.j1 >= 0):
		# 	vm = normalize(joints[self.j1].pos - self.pos)
		# 	f_traction_m = np.dot(joints[self.j1].force, vm)
		# 	if (f_traction_m > 0):
		# 		self.force += 0.9 * f_traction_m * vm
		# if (self.j2 >= 0):
		# 	vp = normalize(joints[self.j2].pos - self.pos)
		# 	f_traction_p = np.dot(joints[self.j2].force, vp)
		# 	if (f_traction_p > 0):
		# 		self.force += 0.9 * f_traction_p * vp



	def update_pos(self, dt):
		if (self.fixed):
			return

		# Do Verlet integration
		self.acc = self.force / self.mass
		self.vel = (self.pos - self.last_pos) / dt
		newPos = 2 * self.pos - self.last_pos + dt * dt * self.acc
		self.last_pos = self.pos
		self.pos = newPos
