import numpy as np

# Define thruster system
class ThrusterSystem:
	def __init__(self, thruster_params):
		self.n_thrusters = thruster_params['n_thrusters']
		self.thrusters = thruster_params["thrusters"]

		self.names = np.array([t['name'] for t in self.thrusters])
		self.positions = np.array([t['position'] for t in self.thrusters])
		self.thrust_limits = np.array([t['thrust_limits'] for t in self.thrusters])
		# self.dead_bands = np.array([t['dead_band'] for t in self.thrusters])
		# 2nd-order actuator model for each thruster: x'' + 2ζωx' + ω²x = ω²u
		self.wn = np.array([t['wn'] for t in self.thrusters])
		self.zeta = np.array([t['zeta'] for t in self.thrusters])

		self.compute_T()
		self.T_inv = self.T.T @ np.linalg.inv(self.T @ self.T.T)
		# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
		# print(self.T)
		
		# States: force and velocity for each thruster
		self.force = np.zeros(6)
		self.thrust = np.zeros(self.n_thrusters)
		self.rpm = np.zeros(self.n_thrusters)

	def compute_T(self):
		self.T = np.zeros((6, self.n_thrusters))
		
		for i in range(self.n_thrusters):
			pos = self.positions[i]
			x, y, z, roll, pitch, yaw = pos
						
			# Rotation matrix ZYX Euler angles
			cr = np.cos(roll); sr = np.sin(roll)
			cp = np.cos(pitch); sp = np.sin(pitch)
			cy = np.cos(yaw); sy = np.sin(yaw)
			
			R = np.array([
				[cy*cp  , cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
				[sy*cp  , sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
				[-sp    , cp*sr           , cp*cr           ]
			])
			
			# Thrust along local x-axis, apply reverse if needed
			F = R @ np.array([1, 0, 0])
			fx, fy, fz = F
			
			# Torque in NED: r x F
			tx = y*fz - z*fy
			ty = z*fx - x*fz
			tz = x*fy - y*fx
			
			self.T[:, i] = [fx, fy, fz, tx, ty, tz]

	def update(self, dt, u_cmd):
		# Compute desired thruster forces using pseudo-inverse control allocation
		thruster_cmd = self.T_inv @ u_cmd
		# Clip desired thruster commands to actuator physical limits
		thruster_cmd = np.clip(thruster_cmd, self.thrust_limits[:, 0], self.thrust_limits[:, 1])

		# Update thruster forces using second-order propeller dynamics
		# Discretize using explicit Euler
		accel = self.wn**2 * (thruster_cmd - self.thrust) - 2 * self.zeta * self.wn * self.rpm
		self.rpm += accel * dt
		self.thrust += self.rpm * dt

		# Thrust limits
		self.thrust = np.clip(self.thrust, self.thrust_limits[:, 0], self.thrust_limits[:, 1])
		# Thrust deadbands
		# dead = (self.thrust >= self.dead_bands[:, 0]) & (self.thrust <= self.dead_bands[:, 1])
		# self.thrust[dead] = 0

		# Clip again to be sure the physical force doesn't exceed limits
		self.force = self.T @ self.thrust
