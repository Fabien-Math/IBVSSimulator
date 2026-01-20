from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from viewer.button import Button

class GUI:
	def __init__(self, window_width=600, window_height=300):
		self.menu_button = Button(10, window_height - 40, 90, 30, "")
		self.draw_reference_button = Button(10, window_height - 230, 20, 20, "Draw reference", active=False)
		self.draw_trace_button = Button(10, window_height - 260, 20, 20, "Draw trace", active=False)
		self.draw_wps_button = Button(10, window_height - 290, 20, 20, "Draw waypoints", active=True)
		self.draw_robot_button = Button(10, window_height - 320, 20, 20, "Draw robot", active=True)
		self.draw_robot_force_button = Button(10, window_height - 350, 20, 20, "Draw robot forces", active=False)
		self.draw_thruster_force_button = Button(10, window_height - 380, 20, 20, "Draw thursts", active=False)

		self.window_width = window_width
		self.window_height = window_height
		self.menu_width = 200


	def draw_hamburger_icon(self, x, y, width=20, line_thickness=2, spacing=4, color=(1, 1, 1)):
		"""
		Draws 3 horizontal lines (hamburger icon) using OpenGL.
		- x, y: bottom-left position (in screen 2D coords)
		- width: width of the lines
		- line_thickness: thickness of each line
		- spacing: space between lines
		"""
		glColor3f(*color)
		for i in range(3):
			y_offset = y + i * (line_thickness + spacing)
			glBegin(GL_QUADS)
			glVertex2f(x, y_offset)
			glVertex2f(x + width, y_offset)
			glVertex2f(x + width, y_offset + line_thickness)
			glVertex2f(x, y_offset + line_thickness)
			glEnd()


	def draw_text(self, x, y, text, font=GLUT_BITMAP_9_BY_15, color=(1, 1, 1)):
		glColor3f(*color)
		glRasterPos2f(x, y)
		for ch in text:
			glutBitmapCharacter(font, ord(ch))

	def draw_rect(self, x, y, w, h, color):
		glColor3f(*color)
		glBegin(GL_QUADS)
		glVertex2f(x, y)
		glVertex2f(x + w, y)
		glVertex2f(x + w, y + h)
		glVertex2f(x, y + h)
		glEnd()

	def update_window_size(self, window_width, window_height):
		self.window_height = window_height
		self.window_width = window_width
		self.draw_reference_button.update_position(10, window_height - 230, 20, 20)
		self.draw_trace_button.update_position(10, window_height - 260, 20, 20)
		self.draw_wps_button.update_position(10, window_height - 290, 20, 20)
		self.draw_robot_button.update_position(10, window_height - 320, 20, 20)
		self.draw_robot_force_button.update_position(10, window_height - 350, 20, 20)
		self.draw_thruster_force_button.update_position(10, window_height - 380, 20, 20)
		self.menu_button.update_position(10, window_height - 40, 90, 30)

	def draw_robot_info(self, eta, nu, line_height=20):
		"""Draws robot state info in aligned format: one column for position/orientation, one for velocity.

		Args:
			eta (list): 6D pose vector [x, y, z, roll, pitch, yaw].
			nu (list): 6D velocity vector [vx, vy, vz, roll_vel, pitch_vel, yaw_vel].
			line_height (int): Vertical spacing between rows.
		"""

		labels = ["x", "y", "z", "R", "P", "Y"]

		# Set column positions
		start_y = self.window_height - 60
		left_col_x = 20
		right_col_x = 120

		for idx, label in enumerate(labels):
			text_pos = f"{label}: {eta[idx]:.2f}"
			text_vel = f"{nu[idx]:.2f}"
			y = start_y - idx * line_height
			self.draw_text(left_col_x, y, text_pos)
			self.draw_text(right_col_x, y, text_vel)

		y = start_y - 6 * line_height
		speed = np.sqrt(nu[0]**2 + nu[1]**2 + nu[2]**2)
		self.draw_text(left_col_x, y, f"speed: {speed:.2f} m/s")


	def draw(self, robot, fps, playback_speed, time):
		glDisable(GL_LIGHTING)
		# Setup 2D orthographic projection
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		gluOrtho2D(0, self.window_width, 0, self.window_height)
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()
		glDisable(GL_DEPTH_TEST)

		if self.menu_button.active:
			self.draw_rect(0, 0, self.menu_width, self.window_height, (0.1, 0.1, 0.1))
			self.draw_robot_info(robot.eta, robot.nu)
			self.draw_reference_button.draw()
			self.draw_trace_button.draw()
			self.draw_wps_button.draw()
			self.draw_robot_button.draw()
			self.draw_robot_force_button.draw()
			self.draw_thruster_force_button.draw()

		self.menu_button.draw()
		self.draw_hamburger_icon(15, self.window_height - 30, width=20)
		self.draw_text(40, self.window_height - 30, "Menu")


		self.draw_text(self.window_width - 10 - len(f"FPS: {fps:.1f}") * 9, self.window_height - 15, f"FPS: {fps:.1f}")
		self.draw_text(self.window_width - 10 - len( f"Speed: {playback_speed:.2f}x") * 9, self.window_height - 30, f"Speed: {playback_speed:.2f}x")
		self.draw_text(self.window_width / 2 - len( f"Simulation time: {time:.2f}s") * 9 / 2, self.window_height - 30, f"Simulation time: {time:.2f}s")

		# Restore state
		glEnable(GL_DEPTH_TEST)
		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)
		glPopMatrix()
		glEnable(GL_LIGHTING)
