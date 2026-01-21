from simulation_parser import load_config
from simulation_manager import SimulationManager
from graph_system import GraphSystem
import time, os
import signal
from numpy import set_printoptions, inf

set_printoptions(threshold=inf, linewidth=inf, formatter={'float': lambda x: "{0:.3e}".format(x)})

def main():
	filename = "config/bluerov_config.yaml"
	# filename = "config/bluerov_config_line.yaml"
	# filename = "config/bluerov_config_cable_tests.yaml"

	launch_time = time.strftime("%Y-%m-%d_%H-%M-%S")

	scenario_params = load_config(filename=filename)

	simulation_params, robot_params, world_params = scenario_params

	sim_manager = SimulationManager(simulation_params, robot_params, world_params)

	# handle CTRL + C
	def handle_sigint(signum, frame):
		sim_manager.viewer.exit = True
	signal.signal(signal.SIGINT, handle_sigint)

	if sim_manager is not None:
		sim_manager.simulate()

	folder = None
	if simulation_params['save_csv'] or simulation_params['save_graphs']:
		folder = simulation_params['save_output'] + '/' + simulation_params['name'] + '_' + launch_time + '/'
		# Create output directory if needed
		os.makedirs(folder, exist_ok=True)
		with open(filename, "r") as fsrc:
			with open(folder + 'config.yaml', "w") as fdst:
				fdst.write(fsrc.read())

	if simulation_params['save_csv']:
		sim_manager.robot.logger.save_to_csv(folder)

	if simulation_params['save_graphs'] or simulation_params['show_graphs']:
		GraphSystem(sim_manager.robot.logger, simulation_params['show_graphs'], simulation_params['save_graphs'], folder)



if __name__ == '__main__':
	main()
