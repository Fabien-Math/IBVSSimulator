import numpy as np
from scipy.spatial.transform import Rotation as R

from logging_system import LoggingSystem

import os
import re

import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

plt.rcParams.update({
		"figure.constrained_layout.use": True,  # auto layout adjustment
		"font.size": 14,                        # base font size
		"axes.titlesize": 16,                   # title font
		"axes.labelsize": 14,                   # axis labels
		"legend.fontsize": 12,                  # legend font
		"xtick.labelsize": 12,                  # tick labels
		"ytick.labelsize": 12,
		"figure.titlesize": 18,                 # suptitle font
		"lines.linewidth": 1.8,                 # slightly thicker lines
		"grid.alpha": 0.4,                      # subtle grid
		"axes.grid": True,                      # grid enabled by default
	})

plt.style.use("seaborn-v0_8")

class GraphSystem:
	def __init__(self, logger, show_graph, save_graph, folder):
		self.logger: LoggingSystem = logger
		self.camera = logger.camera
		self.show_graph = show_graph
		self.save_graph = save_graph
		self.folder = folder

		logger.np_format()

		timestamps = self.logger.timestamps
		etas = self.logger.etas  # actual pose [x, y, z, roll, pitch, yaw]
		etas_err = self.logger.etas - self.logger.etas[-1]
		nus = self.logger.nus    # actual velocities
		visual_error_norm = self.logger.visual_error_norm
		visual_errors = self.logger.visual_errors
		thrust_forces = self.logger.thrust_forces
		thruster_forces = self.logger.thruster_forces
		commands = self.logger.commands
		self.pil_img = None

		self.wanted_markers = self.logger.wanted_markers

		eta_labels = ["X", "Y", "Z", "P", "Q", "R"]
		nu_labels = ["U", "V", "W", "P", "Q", "R"]
		thruster_labels = ["HFR", "HFL", "HRR", "HRL", "VFR", "VFL", "VRR", "VRL"]


		# Plot 1: State variables
		self.plot_dof_grid(timestamps=timestamps, data_actual=etas, labels=eta_labels, title="Pose DoF overview")
		self.plot_dof_grid(timestamps=timestamps, data_actual=nus, labels=nu_labels, title="Velocity DoF overview")

		# Plot 2: Errors
		self.plot_dof_grid(timestamps=timestamps, data_actual=np.where(visual_error_norm > 5, np.nan, visual_error_norm), title=r"Visual Error ($e$)", n=1, m=1)
		self.plot_dof_grid(timestamps=timestamps, data_actual=visual_errors, title=r"Visual Errors ($e_v$)", n=2, m=4)
		self.plot_on_one(timestamps=timestamps, data=visual_errors, title=r"Visual Errors Evolution ($e_v$)", ylabel="Error")
		self.plot_on_one(timestamps=timestamps, data=etas_err, title=r"Pose Errors Evolution ($e_p$)", ylabel="Error")

		# Plot 3: Command
		self.plot_dof_grid(timestamps=timestamps, data_actual=commands, title="Controller Commands")

		# Plot 4: Forces
		self.plot_dof_grid(timestamps=timestamps, data_actual=thrust_forces, title="Thrust Forces (Total)", labels=eta_labels)
		self.plot_dof_grid(timestamps=timestamps, data_actual=thruster_forces, title="Individual Thruster Forces", n=2, m=4, labels=thruster_labels)

		# Plot 5: Thruster forces vs Command
		self.plot_command_vs_thrust(timestamps=timestamps, commands=commands, thrust_forces=thrust_forces)

		# Plot 6: Image graph of the markers positions
		self.pre_image_graph()
		self.plot_image_graph()
		self.post_image_graph()

		if (show_graph):
			plt.show()



	def plot_dof_grid(self, timestamps, data_actual, data_desired=None, data_error=None,
					labels=None, n=2, m=3, title="DoF Overview"):
		"""
		General-purpose function to plot multi-DoF data in an n×m grid.

		Args:
			timestamps (array): Time vector of shape (N,)
			data_actual (array): Actual data (N, D)
			data_desired (array): Desired data (N, D), optional
			data_error (array): Error data (N, D), optional
			labels (list): List of DoF labels of length D
			n (int): Number of subplot rows
			m (int): Number of subplot columns
			title (str): Figure title
		"""
		# --- Validation ---
		if timestamps is None or len(timestamps) == 0:
			print("⚠️ No timestamps provided.")
			return

		timestamps = np.array(timestamps)
		data_actual = np.array(data_actual)
		D = data_actual.shape[1] if data_actual.ndim > 1 else 1

		if labels is None:
			labels = [f"DoF {i+1}" for i in range(D)]

		total_subplots = n * m
		if total_subplots < D:
			print(f"⚠️ Not enough subplots ({total_subplots}) for {D} DoFs. Expanding grid...")
			# Automatically expand the grid
			n = int(np.ceil(D / m))
			total_subplots = n * m
			print(f"→ Adjusted grid: {n} rows × {m} cols")

		# --- Create figure ---
		fig, axes = plt.subplots(n, m, figsize=(5*m, 3.5*n))
		fig.suptitle(title, fontsize=18, fontweight='bold')
		axes = np.atleast_2d(axes)  # ensure 2D array for consistency

		# --- Plot each DoF ---
		for i in range(total_subplots):
			row, col = divmod(i, m)
			ax = axes[row, col]
			if i >= D:
				ax.axis('off')  # hide unused plots
				continue

			# Extract data
			y_actual = data_actual[:, i] if data_actual.ndim > 1 else data_actual
			y_desired = data_desired[:, i] if data_desired is not None and data_desired.ndim > 1 else None
			y_error = data_error[:, i] if data_error is not None and data_error.ndim > 1 else None

			# Plot actual & desired
			ax.plot(timestamps, y_actual, label="Actual", color='tab:blue', lw=1.8)
			if y_desired is not None:
				ax.plot(timestamps, y_desired, '--', label="Desired", color='tab:orange', lw=1.5)

			# Secondary axis for error
			if y_error is not None:
				ax2 = ax.twinx()
				ax2.plot(timestamps, y_error, color='tab:red', alpha=0.6, label="Error")
				ax2.set_ylabel("Error", color='tab:red')
				ax2.tick_params(axis='y', labelcolor='tab:red')
				lines, labels1 = ax.get_legend_handles_labels()
				lines2, labels2 = ax2.get_legend_handles_labels()
				ax.legend(lines + lines2, labels1 + labels2, loc='upper right', fontsize=8)
			else:
				ax.legend(loc='upper right', fontsize=8)

			ax.set_title(labels[i])
			ax.set_xlabel("Time [s]")
			ax.set_ylabel("Value")
			ax.grid(True)

		if self.save_graph:
			self.save_figure(fig, title)



	def plot_command_vs_thrust(self, timestamps, commands, thrust_forces, n_rows=2, n_cols=None):
		"""
		Plot commanded vs actual thrust forces for each thruster.

		Args:
			robot: Robot object with .timestamps, .commands, and .thrust_forces
			n_rows (int): Number of subplot rows (default = 2)
			n_cols (int): Number of subplot columns (auto if None)
		"""
		# --- Validation ---
		if timestamps.size == 0 or commands.size == 0 or thrust_forces.size == 0:
			print("Missing data: timestamps, commands, or thruster forces.")
			return

		n_thrusters = commands.shape[1]
		if thrust_forces.shape[1] != n_thrusters:
			print(f"Mismatch: {thrust_forces.shape[1]} thrust forces for {n_thrusters} commands.")
			return

		# --- Automatic grid calculation ---
		if n_cols is None:
			n_cols = int(np.ceil(n_thrusters / n_rows))

		fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
		fig.suptitle("Thruster Commands vs Actual Thrust Forces", fontsize=18, fontweight='bold')
		axes = np.atleast_2d(axes)
		dofs = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
		for i in range(n_rows * n_cols):
			row, col = divmod(i, n_cols)
			ax = axes[row, col]

			if i >= n_thrusters:
				ax.axis("off")
				continue

			ax.plot(timestamps, commands[:, i], label="Command", color='tab:blue', lw=1.8)
			ax.plot(timestamps, thrust_forces[:, i], label="Thrust", color='tab:orange', lw=1.8)

			ax.set_title(dofs[i])
			ax.set_xlabel("Time [s]")
			ax.set_ylabel("Value [N]")
			ax.grid(True)
			ax.legend(loc='upper right', fontsize=8)


		if self.save_graph:
			self.save_figure(fig, "ThrustersForceVSCommands")



	def save_figure(self, fig, name="plot", file_format="png", dpi=300):
		"""
		Save a matplotlib figure with a timestamped filename.

		Args:
			fig (matplotlib.figure.Figure): The figure to save.
			name (str): Base name for the file.
			file_format (str): File extension ('png', 'jpg', 'pdf', etc.).
			dpi (int): Resolution for raster formats.
		"""
		folder_plot = self.folder + "/plots/"
		# Create output directory if needed
		os.makedirs(folder_plot, exist_ok=True)

		# Generate timestamp
		safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', name)

		# Construct filename
		filename = f"{safe_name}.{file_format}"
		filepath = os.path.join(folder_plot, filename)

		# Save the figure
		fig.savefig(filepath, format=file_format, dpi=dpi, bbox_inches="tight")
		print(f"Figure saved as: {filepath}")

		return filepath

	def save_image(self, img, name="img", file_format="png"):
		"""
		Save a matplotlib figure with a timestamped filename.

		Args:
			img (PIL.Image): The image to save.
			name (str): Base name for the file.
			file_format (str): File extension ('png', 'jpg', 'pdf', etc.).
		"""
		folder_plot = self.folder + "/plots/"
		# Create output directory if needed
		os.makedirs(folder_plot, exist_ok=True)

		# Generate timestamp
		safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', name)

		# Construct filename
		filename = f"{safe_name}.{file_format}"
		filepath = os.path.join(folder_plot, filename)

		cv2.imwrite(filepath, img)
		print(f"Image saved as: {filepath}")

		return filepath

	def pre_image_graph(self):
		self.pil_img = Image.fromarray(self.logger.first_img)

	def post_image_graph(self):
		xm_img = self.wanted_markers[0::2]
		ym_img = self.wanted_markers[1::2]
		for x, y in zip(xm_img, ym_img):
			if x >= -1.0 and x <= 1.0:
				if y >= -1.0 and y <= 1.0:
					draw = ImageDraw.Draw(self.pil_img)

					# compute y and y pos
					x_px = int((x + 1.0)/2 * self.camera.img_width)
					y_px = int((y + 1.0)/2 * self.camera.img_height)

					# Draw the dot
					r = 5
					draw.circle((x_px, y_px), r, width=2, outline=((0, 255, 0)))

		graph_img = np.asarray(self.pil_img)

		if self.save_graph:
			self.save_image(graph_img, "GraphImagePath")


	def plot_image_graph(self):
		if self.pil_img is None:
			return

		for xs, ys in zip(self.logger.x_markers, self.logger.y_markers):
			x_img = xs
			y_img = ys

			for x, y in zip(x_img, y_img):
				if x >= -1.0 and x <= 1.0:
					if y >= -1.0 and y <= 1.0:
						draw = ImageDraw.Draw(self.pil_img)

						# compute y and y pos
						x_px = int((x + 1)/2 * self.camera.img_width)
						y_px = int((y + 1)/2 * self.camera.img_height)

						# Draw the dot
						color=(0, 215, 255)
						r = 0.5
						draw.ellipse((x_px - r, y_px - r, x_px + r, y_px + r), fill=color)




	def plot_on_one(self, timestamps, data, labels=None,
							 title="Multiple Curves vs Time",
							 ylabel="Value"):
		"""
		Plot multiple curves vs time from time-major data.

		Args:
			timestamps (array): Time vector of shape (N,)
			data (array-like): 2D list/array of shape (N_samples, N_curves)
							   [[x(t0), y(t0), ...],
								[x(t1), y(t1), ...]]
			labels (list of str): Labels for each curve
			title (str): Plot title
			ylabel (str): Y-axis label
		"""

		# --- Validation ---
		if timestamps is None or len(timestamps) == 0:
			print("⚠️ No timestamps provided.")
			return
		if data is None or len(data) == 0:
			print("⚠️ No data provided.")
			return

		timestamps = np.asarray(timestamps)
		data = np.asarray(data)

		if data.ndim != 2:
			raise ValueError("Data must be 2D: (N_samples, N_curves)")

		n_samples, n_curves = data.shape

		if timestamps.shape[0] != n_samples:
			raise ValueError(
				f"Timestamps length ({timestamps.shape[0]}) "
				f"does not match data samples ({n_samples})"
			)

		# --- Labels ---
		if labels is None:
			labels = [f"Curve {i+1}" for i in range(n_curves)]

		# --- Style pool (8 distinct styles) ---
		style_pool = [
			{"color": "tab:blue",   "linestyle": "-",  "linewidth": 1.2},
			{"color": "tab:orange", "linestyle": "--", "linewidth": 1.2},
			{"color": "tab:green",  "linestyle": "-.", "linewidth": 1.2},
			{"color": "tab:red",    "linestyle": ":",  "linewidth": 1.6},
			{"color": "tab:purple", "linestyle": "-",  "linewidth": 1.6},
			{"color": "tab:brown",  "linestyle": "--", "linewidth": 1.6},
			{"color": "tab:pink",   "linestyle": "-.", "linewidth": 1.4},
			{"color": "tab:gray",   "linestyle": ":",  "linewidth": 1.4},
		]

		# --- Set styles ---
		styles = [style_pool[i % len(style_pool)] for i in range(n_curves)]

		# --- Plot ---
		fig, ax = plt.subplots(figsize=(8, 4.5))
		ax.set_title(title, fontsize=16, fontweight="bold")

		for i in range(n_curves):
			ax.plot(
				timestamps,
				data[:, i],
				label=labels[i],
				**styles[i]
			)

		ax.set_xlabel("Time [s]")
		ax.set_ylabel(ylabel)
		ax.grid(True)
		ax.legend(fontsize=9)

		if self.save_graph:
			self.save_figure(fig, title)
