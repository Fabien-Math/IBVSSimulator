#!/usr/bin/env python3

from simulation_parser import load_config
from simulation_manager import SimulationManager
from viewer.renderer import Renderer
import os

def main():
    log_folder = "results/waypoints_2025-12-05_11-28-58"  

    print(f"Using folder: {log_folder}")

    config_file = os.path.join(log_folder, "config.yaml")
    if not os.path.isfile(config_file):
        print(f"ERROR: config.yaml not found in folder: {log_folder}")
        return
    
    log_file = os.path.join(log_folder, "output.csv")
    if not os.path.isfile(log_file):
        print(f"ERROR: output.csv not found in folder: {log_folder}")
        return
    
    scenario_params = load_config(config_file)
    simulation_params, robot_params, environment_params = scenario_params

    sim_manager = None
    if simulation_params['graphical']:
        sim_manager = SimulationManager(simulation_params, robot_params, environment_params)

    sim_manager.robot.logger.load_from_csv(log_file)

    viewer = Renderer(sim_manager.robot, sim_manager.dt)
    viewer.run()


if __name__ == "__main__":
    main()
