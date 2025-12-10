import numpy as np


def S(v):
	return np.array([[ 0.00, -v[2],  v[1]],
				     [ v[2],  0.00, -v[0]],
					 [-v[1],  v[0],  0.00]])


class Dynamics:
	def __init__(self, robot_params):		
		# Position and orientation
		self.eta = np.array(robot_params["initial_conditions"]["eta"], dtype=np.float64)
		self.eta_p = np.zeros(6)
		self.eta_prev = self.eta
		self.eta1, self.eta2 = self.eta[0:3], self.eta[3:6]
		self.J1 = np.zeros((3, 3))
		self.J2 = np.zeros((3, 3))
		self.J = np.zeros((6, 6))
		self.J_inv = np.zeros((6, 6))
		self.lastJ = np.zeros((6, 6))
		self.J_p = np.zeros((6, 6))

		# Velocity and angular velocity
		self.nu = np.array(robot_params["initial_conditions"]["nu"], dtype=np.float64)
		self.nu_rel = self.nu
		self.nu1, self.nu2 = self.nu[0:3], self.nu[3:6]
		self.fluid_vel = np.zeros(6)

		# Acceleration and angular acceleration
		self.gamma = np.zeros(6)
		self.forces = np.zeros(6)
		self.hydro_forces = np.zeros(6)
		self.coriolis_centripetal_forces = np.zeros(6)

		# MASS PROPERTIES
		mass_prop = robot_params["mass_properties"]
		self.m = mass_prop["m"]
		self.rg = np.array(mass_prop["rg"])
		self.I0 = np.array(mass_prop["I0"])
		self.Mrb = np.zeros((6, 6))
		self.Mrb[0:3, 0:3] =  self.m * np.identity(3)	# kg
		self.Mrb[0:3, 3:6] = -self.m * S(self.rg)		# kg.m
		self.Mrb[3:6, 0:3] =  self.m * S(self.rg)		# kg.m
		self.Mrb[3:6, 3:6] =  self.I0					# kg.m2
		self.Ma = np.array(mass_prop["Ma"])
		self.M = self.Mrb + self.Ma
		self.M_inv = np.linalg.inv(self.M)
		self.M_eta = np.zeros((6, 6))
		self.M_eta_inv = None

		# DAMPING MATRICES
		damp_prop = robot_params["damping"]
		self.Dl = np.array(damp_prop["Dl"])
		self.Dq = np.array(damp_prop["Dq"])
		self.D_eta = np.zeros((6, 6))

		# CORIOLIS MATRICES
		self.Crb = np.zeros((6, 6))
		self.Ca = np.zeros((6, 6))
		self.C = np.zeros((6, 6))
		self.C_eta = np.zeros((6, 6))

		# VOLUMETRIC FORCES
		self.G = np.zeros(6)
		self.G_eta = np.zeros(6)

		# EXTERNAL FORCES
		self.T = np.zeros(6)
		self.T_eta = np.zeros(6)


		self.time = 0
		self.log = True


	def compute_Crb(self):	# Page 8
		self.Crb[0:3, 3:6] = -self.m * S(self.nu1) - self.m * S(S(self.nu2) @ self.rg)
		self.Crb[3:6, 0:3] = -self.m * S(self.nu1) - self.m * S(S(self.nu2) @ self.rg)
		self.Crb[3:6, 3:6] = -self.m * S(S(self.nu1) @ self.rg) - S(self.I0 @ self.nu2)


	def compute_Ca(self):	# Page 9
		Ma11 = self.Ma[0:3, 0:3]
		Ma12 = self.Ma[0:3, 3:6]
		Ma21 = self.Ma[3:6, 0:3]
		Ma22 = self.Ma[3:6, 3:6]
		self.Ca[0:3, 3:6] = - S(np.matmul(Ma11, self.nu1) + np.matmul(Ma12, self.nu2))
		self.Ca[3:6, 0:3] = - S(np.matmul(Ma11, self.nu1) + np.matmul(Ma12, self.nu2))
		self.Ca[3:6, 3:6] = - S(np.matmul(Ma21, self.nu1) + np.matmul(Ma22, self.nu2))


	def compute_C(self):
		self.compute_Crb()
		self.compute_Ca()
		self.C = self.Crb + self.Ca


	def compute_D(self):
		self.D = self.Dl + np.diag(np.diag(self.Dq) * np.abs(self.nu_rel))


	def compute_T(self, dt):
		self.G = np.zeros(6)
		self.G[2] = -2	# N (Restitution force: gravity + buoyancy)

		self.controller.update(dt, self.eta, self.nu)
		self.T = np.zeros(6)
		self.T += self.thrusters.force


	def compute_gamma(self, dt):
		J_invT = self.J_inv.T
		self.J_p = (self.J - self.lastJ) / dt

		self.M_eta = J_invT @ self.M @ self.J_inv
		self.M_eta_inv = np.linalg.inv(self.M_eta)
		
		self.C_eta = J_invT @ (self.C - self.M @ self.J_inv @ self.J_p) @ self.J_inv
		self.D_eta = J_invT @ self.D @ self.J_inv
		self.G_eta = J_invT @ self.G
		self.T_eta = J_invT @ self.T

		self.hydro_forces = - np.matmul(self.D_eta, self.J @ self.nu_rel)
		self.coriolis_centripetal_forces = - np.matmul(self.C_eta, self.J @ self.nu)
		self.forces = self.coriolis_centripetal_forces + self.hydro_forces + self.T_eta + self.G_eta
		# Update acceleration
		self.gamma = np.matmul(self.M_eta_inv, self.forces)
		# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
		# print(self.forces)


	def compute_J1(self):
		"""
		Rotation matrix from body frame to inertial (NED) frame.
		"""
		sphi = np.sin(self.eta2[0])
		cphi = np.cos(self.eta2[0])
		stheta = np.sin(self.eta2[1])
		ctheta = np.cos(self.eta2[1])
		spsi = np.sin(self.eta2[2])
		cpsi = np.cos(self.eta2[2])

		self.J1 = np.array([
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
		sphi = np.sin(self.eta2[0])
		cphi = np.cos(self.eta2[0])
		ttheta = np.tan(self.eta2[1])
		ctheta = np.cos(self.eta2[1])

		self.J2 =  np.array([
			[1.0,      sphi * ttheta,      cphi * ttheta],
			[0.0,      cphi,               -sphi],
			[0.0,      sphi / ctheta,      cphi / ctheta]
		])
	
	def compute_J(self):
		self.compute_J1()
		self.compute_J2()
		self.J[0:3, 0:3] = self.J1
		self.J[3:6, 3:6] = self.J2
		self.J_inv = np.linalg.inv(self.J)


	def step(self, dt, env):
		self.compute_J()

		# Update relative fluid velocity
		env.compute_fluid_vel(self.eta, dt, self.time)
		self.fluid_vel = env.fluid_vel
		self.nu_rel = self.nu - self.fluid_vel

		# DAMPING
		self.compute_D()
		# CORIOLIS AND CENTRIPETAL
		self.compute_C()
		# EXTERNAL FORCES
		self.compute_T(dt)

		# Acceleration calculation
		self.compute_gamma(dt)

		# Velocity calculation
		self.compute_nu(dt)
		# Position calculation
		self.compute_eta(dt)

		self.time += dt
		self.lastJ = self.J.copy()
	
	def compute_nu(self, dt):
		# Finite difference for speed calculation
		self.eta_p = (self.eta - self.eta_prev) / dt
		# self.eta_p += self.gamma * dt

		self.nu = self.J_inv @ self.eta_p
		self.nu1, self.nu2 = self.nu[0:3], self.nu[3:6]

	def compute_eta(self, dt):
		# Explicit Euler integration for position
		# self.eta += self.eta_p * dt

		# Verlet integration for position
		new_eta = 2*self.eta - self.eta_prev + dt**2 * self.gamma
		self.eta_prev = self.eta
		self.eta = new_eta

		self.eta1, self.eta2 = self.eta[0:3], self.eta[3:6]
