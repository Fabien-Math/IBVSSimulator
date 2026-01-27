import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from scipy.spatial.transform import Rotation as R

from viewer.menu import GUI
from viewer.trace import Trace
from viewer.obj_manager import load_obj_with_tex, create_vertex_data, create_vbo, draw_vbo_textured
import time

RAD2DEG = 180 / 3.1415926535
DEG2RAD = 3.1415926535 / 180
class Renderer:
	def __init__(self, window_width=1600, window_height=900):
		# Viewport variables
		self.window_width = window_width
		self.window_height = window_height

		# Time
		self.time = 0.0
		self.last_frame_time = time.time()
		self.interframe_delay = 16 # millisecond -> 60 FPS
		self.dt = 0.016

		# Playback & animation
		self.running = True
		self.exit = False
		self.step_request = False

		# Camera viewer control
		self.camera_radius = 10
		self.camera_theta = np.pi / 4
		self.camera_phi = np.pi / 4
		self.pan_x, self.pan_y, self.pan_z = 0.0, 0.0, 0.0
		self.mouse_prev = [0, 0]
		self.mouse_button = None

		# Boolean
		self.bool_robot_view = False
		self.bool_follow_robot = False
		self.bool_draw_axis = True

		# Robot related variable (Not initialized yet so the viewer can be run multiple times with different robot)
		self.robot = None
		self.markers = None
		self.marker_colors = None

		# OpenGL resources for fancy robot
		self.robot_vbo = None
		self.robot_texture_id = None
		self.robot_vertex_count = None

		# Other UI/State
		self.gui = None
		self.robot_trace = None

#
	""" ------------------------------------------------------------ """
	""" --------------------   MAIN FUNCTIONS   -------------------- """
	""" ------------------------------------------------------------ """
	def manage_camera_pose(self):
		if self.bool_follow_robot:
			# Pass transform of the object being followed
			tf = self.robot.tf
			pos = tf[:3]
			self.pan_x, self.pan_y, self.pan_z = pos[0], pos[1], pos[2]

		cx = self.camera_radius * np.sin(self.camera_phi) * np.cos(self.camera_theta)
		cy = self.camera_radius * np.sin(self.camera_phi) * np.sin(self.camera_theta)
		cz = self.camera_radius * np.cos(self.camera_phi)

		gluLookAt(cx + self.pan_x, cy + self.pan_y, -cz + self.pan_z,
				self.pan_x, self.pan_y, self.pan_z,
				0, 0, -1)

	def display(self):

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glLoadIdentity()

		self.manage_camera_pose()

		self.draw_ground()

		# Draw robot
		if not self.bool_robot_view:
			self.draw_robot()

		self.draw_markers()

		if self.gui.draw_trace_button.active:
			self.robot_trace.draw()

		if self.gui.draw_reference_button.active:
			self.draw_axis(0.5, 2, True)

		curr_frame_time = time.time()
		fps = 1 / (curr_frame_time - self.last_frame_time)
		self.gui.draw(self.robot, fps, self.dt, self.time)

		self.last_frame_time = curr_frame_time

		glutSwapBuffers()

	def restore_main_viewport(self):
		glViewport(0, 0, self.window_width, self.window_height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(60.0, self.window_width / self.window_height, 0.1, 300.0)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

	def reshape(self, w, h):
		new_width = w if w > 600 else 600
		new_height = h if h > 300 else 300
		self.window_width, self.window_height = new_width, new_height
		self.gui.update_window_size(new_width, new_height)

		glutReshapeWindow(new_width, new_height)
		glViewport(0, 0, new_width, new_height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(60.0, new_width / new_height, 0.1, 1000.0)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()


	def update(self, value):
		if self.running:
			self.time += self.dt
			self.robot.update()

		self.robot_trace.add_last(self.robot.tf[:3])
		glutPostRedisplay()

		glutTimerFunc(self.interframe_delay, self.update, 0)

#
	""" ------------------------------------------------------------ """
	""" --------------------   INIT FUNCTIONS   -------------------- """
	""" ------------------------------------------------------------ """
#
	def init_gl(self):
		# print(glGetString(GL_RENDERER).decode())
		# print(glGetString(GL_VENDOR).decode())
		# print(glGetString(GL_VERSION).decode())
		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION)

		glClearColor(0.7, 0.7, 0.7, 1.0)
		glEnable(GL_DEPTH_TEST)

		# Lighting
		glEnable(GL_LIGHTING)
		glEnable(GL_LIGHT0)

		# Position light above and in front of scene
		light_position = [10.0, 10.0, 10.0, 1.0]  # w=1.0 means positional
		glLightfv(GL_LIGHT0, GL_POSITION, light_position)

		# Light color (white diffuse light)
		glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
		glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])

		# Enable color tracking
		glEnable(GL_COLOR_MATERIAL)
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

		# Optional: Add slight shininess
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32)

		glMatrixMode(GL_PROJECTION)
		gluPerspective(60, self.window_width / self.window_height, 0.1, 1000.0)
		glMatrixMode(GL_MODELVIEW)

	def init_robot(self, robot):
		# Map the robot from the simulation
		self.robot = robot

		# Initialize robot mesh
		robot_mesh = load_obj_with_tex("config/data/BlueROV2H.obj", "config/data/BlueROVTexture.png")
		vertex_data = create_vertex_data(*robot_mesh[:4])
		self.robot_vbo = create_vbo(vertex_data)
		self.robot_texture_id = robot_mesh[4]
		self.robot_vertex_count = len(vertex_data) // 8

		self.robot_trace = Trace()

	def run(self, robot):
		glutInit()
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
		glutInitWindowSize(self.window_width, self.window_height)
		glutCreateWindow(b"OpenGL Viewer")

		self.init_gl()

		glutReshapeFunc(self.reshape)
		glutDisplayFunc(self.display)
		glutTimerFunc(40, self.update, 0)
		glutMouseFunc(self.mouse)
		glutMouseWheelFunc(self.zoom)
		glutMotionFunc(self.motion)
		glutPassiveMotionFunc(self.hover_button)
		glutKeyboardFunc(self.keyboard)
		glutSpecialFunc(self.special_keys)

		# Must be called after OpenGL and GLUT initialized so that the context is well defined
		self.init_robot(robot)

		# GUI
		self.gui = GUI(window_width=self.window_width, window_height=self.window_height)

		glutMainLoop()

#
	""" ------------------------------------------------------------ """
	""" --------------------   DRAW FUNCTIONS   -------------------- """
	""" ------------------------------------------------------------ """
#
	def draw_markers(self, bool_draw_axis=True):
		color = (1, 0, 0)
		for tf in self.robot.tfs:
			glPushMatrix()
			glTranslatef(*tf[:3])
			# glRotatef(tf[5] * RAD2DEG, 0, 0, 1)
			# glRotatef(tf[4] * RAD2DEG, 0, 1, 0)
			# glRotatef(tf[3] * RAD2DEG, 1, 0, 0)
			self.draw_target_marker(color=color)
			if self.bool_draw_axis and bool_draw_axis:
				self.draw_axis(draw_on_top=False)
			glPopMatrix()

	def draw_target_marker(self, size=0.05, color=(1.0, 0.0, 0.0)):
		"""Draws a stylized 3D target marker with sphere and axis arrows."""
		# Draw central sphere
		glDisable(GL_LIGHTING)
		glColor3f(*color)  # red
		glutSolidSphere(size, 20, 20)
		glEnable(GL_LIGHTING)

	def draw_axis(self, length=0.5, line_width=1.5, draw_on_top=False):
		glDisable(GL_LIGHTING)
		if draw_on_top:
			glDisable(GL_DEPTH_TEST)  # Disable depth test to draw on top

		glLineWidth(line_width)  # Set thicker line width
		glBegin(GL_LINES)
		# X axis - red
		glColor3f(1, 0, 0)
		glVertex3f(0, 0, 0)
		glVertex3f(length, 0, 0)
		# Y axis - green
		glColor3f(0, 1, 0)
		glVertex3f(0, 0, 0)
		glVertex3f(0, length, 0)
		# Z axis - blue
		glColor3f(0, 0, 1)
		glVertex3f(0, 0, 0)
		glVertex3f(0, 0, length)
		glEnd()
		glLineWidth(1.0)  # Reset line width

		if draw_on_top:
			glEnable(GL_DEPTH_TEST)  # Re-enable depth test

		glEnable(GL_LIGHTING)

	def draw_robot(self):
		tf = self.robot.tf
		glPushMatrix()
		glTranslatef(*tf[:3])
		glRotatef(tf[5] * RAD2DEG, 0.0, 0.0, 1.0)
		glRotatef(tf[4] * RAD2DEG, 0.0, 1.0, 0.0)
		glRotatef(tf[3] * RAD2DEG, 1.0, 0.0, 0.0)
		draw_vbo_textured(self.robot_vbo, self.robot_vertex_count, self.robot_texture_id)
		if self.bool_draw_axis:
			self.draw_axis()
		glPopMatrix()


	def draw_ground(self, size=100, step=1):
		glDisable(GL_LIGHTING)
		glBegin(GL_LINES)
		glColor3f(0.6, 0.6, 0.6)  # Less reflective grid
		for i in range(-size, size + 1, step):
			glVertex3f(i, -size, 0)
			glVertex3f(i, size, 0)
			glVertex3f(-size, i, 0)
			glVertex3f(size, i, 0)
		glEnd()
		glEnable(GL_LIGHTING)

#
	""" ------------------------------------------------------ """
	""" --------------------   CONTROLS   -------------------- """
	""" ------------------------------------------------------ """
#
	def keyboard(self, key, x, y):
		if key == b' ':
			self.running = not self.running
		elif key in (b'c', b'C'):
			self.bool_robot_view = not self.bool_robot_view
		elif key in (b'f', b'F'):
			self.bool_follow_robot = not self.bool_follow_robot
		elif key in (b'k', b'K'):
			self.camera_phi = np.pi/2
			self.camera_theta = np.pi/2
		elif key in (b'l', b'L'):
			self.camera_phi = 0.0001
			self.camera_theta = np.pi
		elif key in (b'm', b'M'):
			self.gui.menu_button.active = not self.gui.menu_button.active
		elif key in (b'r', b'R'):
			self.gui.draw_reference_button.active = not self.gui.draw_reference_button.active
		elif key in (b's', b'S'):
			self.running = False
			self.step_request = True
		elif key in (b't', b'T'):
			self.gui.draw_trace_button.active = not self.gui.draw_trace_button.active
		elif key == b'\x1b':  # ESC
			print("Exiting simulation...")
			glutLeaveMainLoop()

	def special_keys(self, key, x, y):
		if key == GLUT_KEY_LEFT:
			self.dt = np.sign(self.dt) * max(1/2**6, np.abs(self.dt) / 2)
		elif key == GLUT_KEY_DOWN:
			self.dt = 0.01
		elif key == GLUT_KEY_UP:
			self.dt = 0.01
		elif key == GLUT_KEY_RIGHT:
			self.dt = np.sign(self.dt) * min(2**6, np.abs(self.dt) * 2)

	def mouse(self, button, state, x, y):
		if state == GLUT_DOWN:
			self.mouse_button = button
			self.mouse_prev = [x, y]
		else:
			self.mouse_button = None

		y = self.window_height - y  # Invert Y for UI coords

		# Toggle menu button
		self.gui.menu_button.handle_mouse(state)

		if state != GLUT_DOWN:
			return

		if not self.gui.menu_button.active:
			return

		self.gui.draw_reference_button.handle_mouse(state)
		self.gui.draw_trace_button.handle_mouse(state)
		self.gui.draw_wps_button.handle_mouse(state)
		self.gui.draw_robot_button.handle_mouse(state)
		self.gui.draw_robot_force_button.handle_mouse(state)
		self.gui.draw_thruster_force_button.handle_mouse(state)

	def hover_button(self, x, y):
		y = self.window_height - y  # Invert Y for UI coords
		self.gui.menu_button.handle_mouse_motion(x, y)

		if not self.gui.menu_button.active:
			return
		self.gui.draw_reference_button.handle_mouse_motion(x, y)
		self.gui.draw_trace_button.handle_mouse_motion(x, y)
		self.gui.draw_wps_button.handle_mouse_motion(x, y)
		self.gui.draw_robot_button.handle_mouse_motion(x, y)
		self.gui.draw_robot_force_button.handle_mouse_motion(x, y)
		self.gui.draw_thruster_force_button.handle_mouse_motion(x, y)

	def zoom(self, wheel, direction, x, y):
		self.camera_radius *= 0.9 if direction > 0 else 1.1
		self.camera_radius = np.clip(self.camera_radius, 0.2, 1000.0)

	def motion(self, x, y):
		dx = x - self.mouse_prev[0]
		dy = y - self.mouse_prev[1]
		self.mouse_prev = [x, y]

		if self.mouse_button == GLUT_RIGHT_BUTTON:
			self.camera_theta += dx * 0.005
			self.camera_phi -= dy * 0.005
			self.camera_phi = np.clip(self.camera_phi, 0.01, np.pi - 0.01)
		elif self.mouse_button == GLUT_MIDDLE_BUTTON:
			cam_x = self.camera_radius * np.sin(self.camera_phi) * np.cos(self.camera_theta)
			cam_y = self.camera_radius * np.sin(self.camera_phi) * np.sin(self.camera_theta)
			cam_z = self.camera_radius * np.cos(self.camera_phi)

			forward = np.array([-cam_x, -cam_y, cam_z])
			forward /= np.linalg.norm(forward)
			up = np.array([0, 0, -1])
			right = np.cross(forward, up)
			right /= np.linalg.norm(right)
			true_up = np.cross(right, forward)

			factor = 0.001 * self.camera_radius
			move = right * (-dx * factor) + true_up * (dy * factor)

			self.pan_x += move[0]
			self.pan_y += move[1]
			self.pan_z += move[2]
