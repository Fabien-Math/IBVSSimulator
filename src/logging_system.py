import numpy as np
import os

class LoggingSystem:
	def __init__(self, robot):
		self.launch_time = None
		self.robot = robot
		self.camera = robot.camera
		self.first_img = None

		# Time
		self.timestamps = []

		# State
		self.etas = []
		self.nus = []
		self.timestamps = []

		# Fluid velocity
		self.fluid_vels = []

		# Forces
		self.thrust_forces = []
		self.thruster_forces = []
		self.hydro_forces = []
		self.forces = []

		# Command
		self.commands = []

		# Errors
		self.visual_error_norm = []
		self.visual_errors = []

		# Markers
		self.wanted_markers = robot.controller.wanted_marker_pos
		self.x_markers = []
		self.y_markers = []


		self.errs = False
		self.thrust_force = False

	def log_state(self, timestamp):
		"""Append the current robot state to logs."""
		# Time
		self.timestamps.append(timestamp)

		# State
		self.etas.append(np.array(self.robot.eta))
		self.nus.append(np.array(self.robot.nu))

		# Forces
		if self.robot.thrusters is not None:
			self.thrust_forces.append(np.array(self.robot.thrusters.force))
			self.thruster_forces.append(np.array(self.robot.thrusters.thrust))
		self.hydro_forces.append(np.array(self.robot.dynamics.hydro_forces))
		self.forces.append(np.array(self.robot.dynamics.forces))

		# Fluid velocity
		self.fluid_vels.append(np.array(self.robot.fluid_vel))

		# Command
		if self.robot.controller is not None:
			self.commands.append(self.robot.controller.cmd)

		self.visual_error_norm.append([self.robot.controller.error_norm])
		self.visual_errors.append(np.array(self.robot.controller.errors))

		# Markers
		self.x_markers.append(np.array(self.robot.controller.xs))
		self.y_markers.append(np.array(self.robot.controller.ys))



	def clear_logs(self):
		"""Clear all logs."""
		# Time
		self.timestamps.clear()

		# State
		self.etas.clear()
		self.nus.clear()

		# Desired state
		self.etas_desired.clear()
		self.nus_desired.clear()

		# Fluid velocity
		self.fluid_vels.clear()

		# Forces
		self.thrust_forces.clear()
		self.thruster_forces.clear()
		self.hydro_forces.clear()
		self.forces.clear()

		# Command
		self.commands.clear()

		# Errors
		self.visual_error_norm.clear()
		self.visual_errors.clear()

		# Markers
		self.x_markers.clear()
		self.y_markers.clear()


	def np_format(self):
		# Time
		self.timestamps = np.array(self.timestamps)

		# State
		self.etas = np.array(self.etas)
		self.nus = np.array(self.nus)

		# Fluid velocity
		self.fluid_vels = np.array(self.fluid_vels)

		# Forces
		self.thrust_forces = np.array(self.thrust_forces)
		self.thruster_forces = np.array(self.thruster_forces)
		self.hydro_forces = np.array(self.hydro_forces)
		self.forces = np.array(self.forces)

		# Command
		self.commands = np.array(self.commands)

		# Errors
		self.visual_error_norm = np.array(self.visual_error_norm)
		self.visual_errors = np.array(self.visual_errors)

		# Markers
		self.x_markers = np.array(self.x_markers)
		self.y_markers = np.array(self.y_markers)


	def save_to_csv(self, folder, number_format="{:.6e}"):
		"""
		Save all numerical data to CSV using a consistent number format.
		number_format: Python string format, e.g. "{:.3f}", "{:.2e}", etc.
		"""
		import csv
		import os

		# Helper: formats numbers or returns non-numbers unchanged
		def fmt(x):
			if isinstance(x, (int, float)):
				return number_format.format(x)
			return x

		filename = "output.csv"
		filepath = os.path.join(folder, filename)

		with open(filepath, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)

			# ----- Build header -----
			header = ['timestamp']

			# Utility to extend header lists
			def add_header(prefix, data):
				if len(data) and len(data[0]) > 0:
					header.extend([f"{prefix}{i}" for i in range(len(data[0]))])

			add_header("eta_", self.etas)
			add_header("nu_", self.nus)
			add_header("fluid_vel_", self.fluid_vels)

			add_header("thrust_force_", self.thrust_forces)
			add_header("thruster_force_", self.thruster_forces)
			add_header("hydro_force_", self.hydro_forces)
			add_header("force_", self.forces)

			add_header("command_", self.commands)
			add_header("visual_error_norm_", self.visual_error_norm)
			add_header("visual_error_", self.visual_errors)

			add_header("x_markers_", self.x_markers)
			add_header("y_markers_", self.y_markers)

			writer.writerow(header)

			# ----- Write rows -----
			for i in range(len(self.timestamps)):
				row = [fmt(self.timestamps[i])]

				def add_row(data):
					if data is not None and i < len(data) and data[i] is not None:
						return [fmt(x) for x in data[i]]
					return []

				row += add_row(self.etas)
				row += add_row(self.nus)
				row += add_row(self.fluid_vels)

				row += add_row(self.thrust_forces)
				row += add_row(self.thruster_forces)
				row += add_row(self.hydro_forces)
				row += add_row(self.forces)

				row += add_row(self.commands)
				row += add_row(self.visual_error_norm)
				row += add_row(self.visual_errors)

				row += add_row(self.x_markers)
				row += add_row(self.y_markers)

				writer.writerow(row)

		print("Simulation saved!")


	def load_from_csv(self, filepath):
		"""
		Load logged CSV data and reconstruct all variables
		that exist in the file. Variables not present remain unchanged.
		"""

		import csv

		# ========== Read CSV into a dict of columns ==========
		with open(filepath, 'r', newline='') as csvfile:
			reader = csv.reader(csvfile)
			header = next(reader)

			columns = {name: [] for name in header}

			for row in reader:
				for name, value in zip(header, row):
					try:
						value = float(value)
					except ValueError:
						pass
					columns[name].append(value)

		# ========== Helper: extract vector groups ==========
		import re
		def extract_group(prefix):
			"""
			Returns a list-of-lists for all columns matching prefix + index.
			Only accepts keys like prefix + digits (e.g. 'eta_0').
			"""
			numeric_keys = []
			for k in columns:
				if k.startswith(prefix):
					suffix = k[len(prefix):]
					if suffix.isdigit():  # <-- strict check
						numeric_keys.append(k)

			# Nothing found
			if not numeric_keys:
				return None

			# Sort by numeric index
			numeric_keys = sorted(numeric_keys, key=lambda k: int(k[len(prefix):]))

			# Build rows
			rows = len(columns[numeric_keys[0]])
			group = []

			for i in range(rows):
				group.append([columns[k][i] for k in numeric_keys])

			return group


		# ========== Recover variables if present ==========

		# Timestamp (scalar column)
		if "timestamp" in columns:
			self.timestamps = columns["timestamp"]

		# State
		self.etas = extract_group("eta_") or self.etas
		self.nus = extract_group("nu_") or self.nus

		# Fluid velocity
		self.fluid_vels = extract_group("fluid_vel_") or self.fluid_vels

		# Forces
		self.thrust_forces = extract_group("thrust_force_") or self.thrust_forces
		self.thruster_forces = extract_group("thruster_force_") or self.thruster_forces
		self.hydro_forces = extract_group("hydro_force_") or self.hydro_forces
		self.forces = extract_group("force_") or self.forces

		# Commands
		self.commands = extract_group("command_") or self.commands

		# Errors
		self.visual_error_norm = extract_group("visual_error_norm_") or self.visual_error_norm
		self.visual_errors = extract_group("visual_errors_") or self.visual_errors

		# Markers
		self.x_markers = extract_group("x_markers_") or self.x_markers
		self.y_markers = extract_group("y_markers_") or self.y_markers

		self.np_format()

		print("Simulation log loaded!")
