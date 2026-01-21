import numpy as np
from scipy.spatial.transform import Rotation as R

# Robot components
from thruster_system import ThrusterSystem
from logging_system import LoggingSystem
from dynamics import Dynamics
from camera import Camera
# Controller
from controllers.ibvs_controller import IBVSController
from controllers.ibvs_smc_controller import IBVSSMController
from controllers.ibvs_pid_controller import IBVSPIDController
from controllers.ibvs_adaptative_controller import IBVSADAPController


class Robot:
	def __init__(self, robot_params, world):
		self.world = world

		# Robot time
		self.time = 0

		# Position and orientation
		self.eta = np.array(robot_params["initial_conditions"]["eta"], dtype=np.float64)
		self.eta_prev = self.eta

		# Velocity and angular velocity
		self.nu = np.array(robot_params["initial_conditions"]["nu"], dtype=np.float64)
		self.fluid_vel = np.zeros(6)

		# Robot components
		self.dynamics = Dynamics(robot_params, world)
		self.thrusters = ThrusterSystem(robot_params["thruster"])

		# IBVS controller
		self.mission_finished = False
		self.kinematics_only = robot_params["mission"]["kinematics_only"]
		self.camera = Camera(robot_params["camera"])
		ctrl_type = robot_params["controller"]['type']
		match ctrl_type:
			case 'SIMPLE':
				self.controller = IBVSController(robot_params["controller"], robot_params["mission"], self.camera, world)
			case 'ADAP':
				self.controller = IBVSADAPController(robot_params["controller"], robot_params["mission"], self.camera, world)
			case 'SMC':
				self.controller = IBVSSMController(robot_params["controller"], robot_params["mission"], self.camera, world)
			case 'PID':
				self.controller = IBVSPIDController(robot_params["controller"], robot_params["mission"], self.camera, world)
			case _:
				raise ValueError(f"Unsupported controller type: {ctrl_type}")


		# Robot logger
		self.logger = LoggingSystem(self)
		self.first_img = True

	def update(self, dt):
		self.time += dt

		# Update environment
		self.world.update(self.eta, dt, self.time)
		self.fluid_vel = self.world.ocean.fluid_vel

		# Update robot components
		self.dynamics.update_transformation_matrices()
		self.controller.update(self.eta, self.nu, dt, self.time)
		if self.controller.error_norm < self.controller.tolerance and np.linalg.norm(self.nu) < 0.01:
			self.mission_finished = True

		self.thrusters.update(dt, self.controller.cmd + np.array([0, 0, 2, 0, 0, 0]))  # Apply thruster dynamics
		self.dynamics.step(dt, self.eta, self.eta_prev, self.nu, self.thrusters.force, self.fluid_vel)
		if self.kinematics_only:
			self.eta += dt * self.dynamics.J @ self.controller.cmd
		else:
			self.eta = self.dynamics.eta.copy()
			self.eta_prev = self.dynamics.eta_prev.copy()
			self.nu = self.dynamics.nu.copy()

		# Finnaly log the robot state
		self.logger.log_state(self.time)

	def update_img(self, img):
		self.controller.update_img(img)

		if self.first_img:
			self.logger.first_img = img.copy()
			self.first_img = False
