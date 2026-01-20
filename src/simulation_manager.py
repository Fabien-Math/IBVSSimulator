from ocean import Ocean
from robot import Robot
from world import World
from viewer.renderer import Renderer

class SimulationManager:
	def __init__(self, simulation_params, robot_params, environment_params, world_params):
		self.dt = simulation_params['timestep']
		self.end_time = simulation_params['end_time']

		self.ocean = Ocean(environment_params)
		self.world = World(world_params)
		self.robot = Robot(robot_params)
		self.viewer = Renderer(self.dt, self.end_time, simulation_params['graphical'], simulation_params["fancy_robot"])

		self.robot.controller.init_world(self.world)


	def simulate(self):

		self.viewer.run(self.robot, self.world, self.ocean)

		if self.robot.mission_finished:
			print(f"Mission accomplished!\nTotal simulation time: {self.viewer.time:.3f} s")
		else:
			print(f"Mission NOT finished!\nTotal simulation time: {self.viewer.time:.3f} s")

		self.robot.logger.np_format()
