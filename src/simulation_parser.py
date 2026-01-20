import yaml
import numpy as np
import re

def load_config(filename):
	"""
	Load full config including robot, thruster, controller, and current parameters from YAML.
	Returns dictionaries for each section with numpy arrays where applicable.
	"""
	with open(filename, 'r') as f:
		cfg = yaml.safe_load(f)

	simulation_cfg = cfg.get('simulation', {})
	simulation_params = {
		'name': simulation_cfg.get('name', "simulation"),
		'timestep': simulation_cfg.get('timestep', 0.01),
		'end_time': simulation_cfg.get('end_time', 10.0),
		'graphical': simulation_cfg.get('graphical', True),
		'fancy_robot': simulation_cfg.get('fancy_robot', True),
		'save_csv': simulation_cfg.get('save_csv', True),
		'save_output': simulation_cfg.get('save_output', ""),
		'show_graphs': simulation_cfg.get('show_graphs', False),
		'save_graphs': simulation_cfg.get('save_graphs', False),
	}

	# Robot parameters
	robot_cfg = cfg.get('robot', {})
	# Thruster parameters
	thrusters_cfg = robot_cfg.get('thrusters', {})
	n_thursters = int(thrusters_cfg.get('n_thrusters'))

	thrusters = [{"name": "", "position": None, "thrust_limits": None, "wn": 0, "zeta": 0}
			  		for _ in range(n_thursters)]
	# thrusters = [{"name": "", "position": None, "thrust_limits": None, "dead_band": None, "wn": 0, "zeta": 0}
	# 		  		for _ in range(n_thursters)]
	for i in range(n_thursters):
		thruster_cfg = thrusters_cfg.get('thruster'+ str(i), {})
		thrusters[i]["name"] = thruster_cfg.get('name', "thruster" + str(i))
		thrusters[i]["position"] = np.array(thruster_cfg.get('position', None))
		thrusters[i]["thrust_limits"] = np.array(thruster_cfg.get('thrust_limits', np.array([-1e6, 1e6])))
		# thrusters[i]["dead_band"] = np.array(thruster_cfg.get('dead_band', np.zeros(2)))
		thrusters[i]["wn"] = np.array(thruster_cfg.get('wn', 0))
		thrusters[i]["zeta"] = np.array(thruster_cfg.get('zeta', 0))

	thruster_params = {
		'n_thrusters': n_thursters,
		'thrusters': thrusters,
	}

	mission_cfg = robot_cfg.get('mission', {})
	mission = {
		'type': mission_cfg.get('type'),
		'marker_pos_des': np.array(mission_cfg.get('marker_position_in_image'), dtype=float),
		'tolerance': mission_cfg.get('error_tol', 0.0),
		'lambda': mission_cfg.get('lambda', 0.0),
		'ratio_zs': mission_cfg.get('ratio_zs', 0.0),
		'img_computation': mission_cfg.get('img_computation', False),
		'save_images': simulation_cfg.get('save_images', False),
	}

	camera_cfg = robot_cfg.get('camera', {})
	camera = {
		'tf': np.array(camera_cfg.get('tf'), dtype=float),
		'img_width': camera_cfg.get('img_width', 0),
		'img_height': camera_cfg.get('img_height', 0),
		'fov': camera_cfg.get('fov', 0),
		'fps': camera_cfg.get('fps', 0),
	}

	robot_params = {
		'name': robot_cfg.get('name', 'unknown'),
		'mission': mission,
		'camera': camera,
		'initial_conditions': {
			'eta': np.array(robot_cfg.get('initial_conditions', {}).get('eta')),
			'nu': np.array(robot_cfg.get('initial_conditions', {}).get('nu'))
		},
		'mass_properties': {
			'm': robot_cfg.get('mass_properties', {}).get('m', 0.0),
			'rg': np.array(robot_cfg.get('mass_properties', {}).get('rg')),
			'I0': np.array(robot_cfg.get('mass_properties', {}).get('I0')),
			'Ma': np.array(robot_cfg.get('mass_properties', {}).get('Ma')),
		},
		'damping': {
			'Dl': np.array(robot_cfg.get('damping', {}).get('Dl')),
			'Dq': np.array(robot_cfg.get('damping', {}).get('Dq')),
		},
		'thruster': thruster_params,
	}


	# World parameters
	world_cfg = cfg.get('world', {})

	# Environment parameters
	ocean_cfg = world_cfg.get('ocean', {})
	current_cfg = ocean_cfg.get('current', {})
	current_types = re.sub(r"\s+", "", current_cfg.get('types', 'zero'))
	current_params = {'types': current_types}

	for current_type in current_types.split(','):
		if current_type == 'normal':
			current_params['normal'] = {}
			current_params['normal']['speed'] = np.array(current_cfg.get('normal', {}).get('speed'))
			current_params['normal']['std'] = np.array(current_cfg.get('normal', {}).get('std'))
		elif current_type == 'jet':
			current_params['jet'] = {}
			current_params['jet']['vector'] = np.array(current_cfg.get('jet', {}).get('vector'))
			current_params['jet']['period'] = current_cfg.get('jet', {}).get('period')
			current_params['jet']['duty'] = current_cfg.get('jet', {}).get('duty')
		elif current_type == 'constant':
			current_params['constant'] = {}
			current_params['constant']['vector'] = np.array(current_cfg.get('constant', {}).get('vector'))
		elif current_type == 'time_series':
			current_params['time_series'] = [
				{'time': entry.get('time', 0), 'vector': np.array(entry.get('vector'))}
				for entry in current_cfg.get('time_series')
			]
		elif current_type == 'depth_profile':
			current_params['depth_profile'] = [
				{'depth': entry.get('depth', 0), 'vector': np.array(entry.get('vector'))}
				for entry in current_cfg.get('depth_profile')
			]
		elif current_type == 'zero':
			pass
		else:
			raise ValueError(f"Unsupported current type: {current_type}")

	properties_cfg = ocean_cfg.get('properties', {})
	ocean_params = {
		'water_density': properties_cfg.get('water_density', 1025.0),
		'water_viscosity': properties_cfg.get('water_viscosity', 1.0e-3),
		'current': current_params,
	}


	markers_cfg = world_cfg.get('markers', {})
	cables_cfg = world_cfg.get('cables', {})
	marker_params = {
		'initial_position': np.array(markers_cfg.get('initial_position'), dtype=float),
		'translation': np.array(markers_cfg.get('translation', 0.01), dtype=float),
		'rotation': np.array(markers_cfg.get('rotation', 10.0), dtype=float),
		'colors': np.array(markers_cfg.get('colors'), dtype=float),
	}

	cables_params = []

	cable_nb = int(cables_cfg.get('cable_nb', 0))
	for i in range(cable_nb):
		cable_cfg = cables_cfg.get(f'cable{i}', {})
		cable_params = {
			'initial_position': np.array(cable_cfg.get('initial_position', [0.0, 0.0, 0.0]), dtype=float),
			'end_position': np.array(cable_cfg.get('end_position', [0.0, 0.0, 0.0]), dtype=float),
			'length': float(cable_cfg.get('length', 0.0)),
			'radius': float(cable_cfg.get('radius', 0.0)),
			'n_subdiv': int(cable_cfg.get('n_subdiv', 0)),
			'linear_mass': float(cable_cfg.get('linear_mass', 0.0)),
			'bending_coef': float(cable_cfg.get('bending_coef', 0.0)),
			'jakobsen_params': {
				'n_iter': int(cable_cfg.get('jakobsen_params', {}).get('n_iter', 0)),
				'penalty_coef': float(cable_cfg.get('jakobsen_params', {}).get('penalty_coef', 0.0)),
			},
			'anchored': np.array(cable_cfg.get('anchored', [0, 0]), dtype=int),
			'color': np.array(cable_cfg.get('color', [1.0, 1.0, 1.0]), dtype=float),
		}
		cables_params.append(cable_params)


	world_params = {
		'gravity': np.array(properties_cfg.get('gravity', [0.0, 0.0, 9.81])),
		'ocean': ocean_params,
		'markers': marker_params,
		'cables':cables_params
	}


	return simulation_params, robot_params, world_params
