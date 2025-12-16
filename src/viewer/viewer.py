import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from scipy.spatial.transform import Rotation as R

from viewer.obj_manager import load_obj_with_tex, create_vertex_data, create_vbo, draw_vbo_textured
from viewer.offscreen_renderer import OffscreenRenderer

RAD2DEG = 180 / 3.1415926535
DEG2RAD = 3.1415926535 / 180
class Viewer:
	def __init__(self, timestep, window_width=1600, window_height=900):
		# Viewport variables
		self.window_width = window_width
		self.window_height = window_height

		# Time
		self.dt = timestep
		self.time = 0.0

		# Playback & animation
		self.running = True
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
		self.bool_follow_robot = True
		self.bool_draw_axis = True
		self.bool_draw_fancy_robot = False
		self.bool_draw_wanted_markers = False

		# Robot related variable (Not initialized yet so the viewer can be run multiple times with different robot)
		self.robot = None
		self.markers = None
		self.markers_color = None
		self.wanted_markers = None
		# Robot camera
		self.cam_width = None
		self.cam_height = None
		self.cam_fov = None
		self.cam_fps = None
		self.offscreen_renderer = None
		self.img = None
		self.last_img_ts = 0.0

		# OpenGL resources for fancy robot
		self.robot_vbo = None
		self.robot_texture_id = None
		self.robot_vertex_count = None


	""" ------------------------------------------------------------ """
	""" --------------------   MAIN FUNCTIONS   -------------------- """
	""" ------------------------------------------------------------ """

	def manage_camera_pose(self):
		if self.bool_robot_view:
			cx, cy, cz = self.robot.eta[:3]

			V = np.array([cx, cy, cz + 1000])
			Rwr = R.from_euler('xyz', self.robot.eta[3:]).as_matrix()
			Rrc = self.robot.Rrc
			self.pan_x, self.pan_y, self.pan_z = Rwr @ Rrc @ V

			up_local = np.array([0, -1, 0])
			up = Rwr @ Rrc @ up_local

			gluLookAt(cx, cy, cz,
					self.pan_x, self.pan_y, self.pan_z,
					up[0], up[1], up[2])

		else:
			if self.bool_follow_robot:
				tf = self.robot.eta
				pos = tf[:3]
				self.pan_x, self.pan_y, self.pan_z = pos[0], pos[1], pos[2]

			cx = self.camera_radius * np.sin(self.camera_phi) * np.cos(self.camera_theta)
			cy = self.camera_radius * np.sin(self.camera_phi) * np.sin(self.camera_theta)
			cz = self.camera_radius * np.cos(self.camera_phi)

			gluLookAt(cx + self.pan_x, cy + self.pan_y, -cz + self.pan_z,
					self.pan_x, self.pan_y, self.pan_z,
					0, 0, -1)

	def render_onboard_camera(self):

		if self.img is not None:
			if self.time < self.last_img_ts + 1 / self.cam_fps:
				return self.img

		# Bind the offscreen FBO
		self.offscreen_renderer.bind()

		# Clear buffers
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		# Set projection
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(self.cam_fov, self.cam_width / self.cam_height, 0.01, 300.0)

		# Set modelview / camera
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		cx, cy, cz = self.robot.eta[:3]
		V = np.array([cx, cy, cz + 1000])
		Rwr = R.from_euler('xyz', self.robot.eta[3:]).as_matrix()
		Rrc = self.robot.Rrc
		pan_x, pan_y, pan_z = Rwr @ Rrc @ V

		up_local = np.array([0, -1, 0])
		up = Rwr @ Rrc @ up_local

		gluLookAt(cx, cy, cz,
				pan_x, pan_y, pan_z,
				up[0], up[1], up[2])

		# Draw the scene (markers, etc.)
		self.draw_markers(bool_draw_axis=False)

		# Flush and read pixels
		glFlush()

		self.img = self.offscreen_renderer.read_pixels()
		self.last_img_ts = self.time

		# Unbind the FBO to restore main framebuffer
		self.offscreen_renderer.unbind()
		self.restore_main_viewport()

		return self.img

	def display(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glLoadIdentity()
		
		self.manage_camera_pose()

		self.draw_ground()
		if not self.bool_robot_view:
			if self.bool_draw_fancy_robot:
				self.draw_robot()
			else:
				self.draw_cubic_robot()
		else:
			if self.bool_draw_wanted_markers:
				self.draw_marker_positions()

		self.draw_markers()

		if self.bool_draw_axis:
			self.draw_axis(0.5, 2, True)

		glutSwapBuffers()

	def update(self, value):
		self.dt = self.robot.dt

		if self.running or self.step_request:
			self.time += self.dt

			img = self.render_onboard_camera()
			self.robot.update_visual_servoing(img)
			self.step_request = False

		glutPostRedisplay()
		glutTimerFunc(40, self.update, 0)

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

		glutReshapeWindow(new_width, new_height)
		glViewport(0, 0, new_width, new_height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(60.0, new_width / new_height, 0.1, 300.0)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()



	""" ------------------------------------------------------------ """
	""" --------------------   INIT FUNCTIONS   -------------------- """
	""" ------------------------------------------------------------ """

	def init_gl(self):
		# print(glGetString(GL_RENDERER).decode())
		# print(glGetString(GL_VENDOR).decode())
		# print(glGetString(GL_VERSION).decode())

		glClearColor(0.1, 0.1, 0.1, 1.0)
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
		self.markers = robot.markers
		self.wanted_markers = robot.wanted_marker_pos
		self.markers_color = robot.markers_color

		# Update robot camera info
		self.cam_width = robot.cam_width
		self.cam_height = robot.cam_height
		self.cam_fov = robot.cam_fov
		self.cam_fps = robot.cam_fps
		self.last_img_ts = 0.0

		# Load models
		if self.bool_draw_fancy_robot:
			robot_mesh = load_obj_with_tex("data/BlueROV2H.obj", "data/BlueROVTexture.png")
			vertex_data = create_vertex_data(*robot_mesh[:4])
			self.robot_vbo = create_vbo(vertex_data)
			self.robot_texture_id = robot_mesh[4]
			self.robot_vertex_count = len(vertex_data) // 8

		self.offscreen_renderer = OffscreenRenderer(self.cam_width, self.cam_height)
	
	def run(self, robot):
		# Initialize GLUT viewport
		glutInit()
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
		glutInitWindowSize(self.window_width, self.window_height)
		glutCreateWindow(b"3D Robot Viewer")

		# Initialize OpenGL
		self.init_gl()

		# Initialize all GLUT functions and callbacks
		glutReshapeFunc(self.reshape)
		glutDisplayFunc(self.display)
		glutTimerFunc(40, self.update, 0)
		glutMouseFunc(self.mouse)
		glutMouseWheelFunc(self.zoom)
		glutMotionFunc(self.motion)
		glutKeyboardFunc(self.keyboard)
		glutSpecialFunc(self.special_keys)
		glutJoystickFunc(self.joystick_func, 50)		
		
		# Must be called after OpenGL and GLUT initialized
		self.init_robot(robot)
		
		glutMainLoop()



	""" ------------------------------------------------------------ """
	""" --------------------   DRAW FUNCTIONS   -------------------- """
	""" ------------------------------------------------------------ """

	def draw_circle_outline(self, x, y, radius=10, segments=64):
		"""Draw an empty circle (outline) at (x, y)."""
		glBegin(GL_LINE_LOOP)
		for i in range(segments):
			theta = 2.0 * np.pi * i / segments
			glVertex2f(x + radius * np.cos(theta),
					y + radius * np.sin(theta))
		glEnd()

	def draw_markers(self, bool_draw_axis=True):
		for tf, color in zip(self.markers, self.markers_color):
			glPushMatrix()
			glTranslatef(*tf)
			# glRotatef(tf[5] * RAD2DEG, 0, 0, 1)
			# glRotatef(tf[4] * RAD2DEG, 0, 1, 0)
			# glRotatef(tf[3] * RAD2DEG, 1, 0, 0)
			self.draw_target_marker(color=color)
			if self.bool_draw_axis and bool_draw_axis:
				self.draw_axis(draw_on_top=False)
			glPopMatrix()

	def draw_marker_positions(self):
		"""
		Draw a round HUD panel in the bottom-right showing the current vector.
		"""
		# ---- Switch to 2D mode ----
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		glOrtho(0, self.window_width, 0, self.window_height, -1, 1)

		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()

		glDisable(GL_LIGHTING)
		glDisable(GL_DEPTH_TEST)

		# --- Enable blending / line smoothing ---
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable(GL_LINE_SMOOTH)
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

		# ------------------------------------------
		# Draw outline
		# ------------------------------------------
		glColor3f(0.90, 0.90, 0.90)
		glLineWidth(2)
		for pos in self.wanted_markers:
			cx, cy = pos
			zoom_x = self.window_width / self.cam_width 
			zoom_y = self.window_height / self.cam_height 
			glBegin(GL_LINE_LOOP)
			for i in range(0, 361, 3):
				ang = np.radians(i)
				glVertex2f(cx * zoom_x + np.cos(ang) * 10,
						cy * zoom_y + np.sin(ang) * 10)
			glEnd()
		glLineWidth(1)

		# ---- Restore state ----
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_LIGHTING)

		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)
		glPopMatrix()

	def draw_target_marker(self, size=0.05, color=(1.0, 0.0, 0.0)):
		"""Draws a stylized 3D target marker with sphere and axis arrows."""
		# Draw central sphere
		glDisable(GL_LIGHTING)
		glColor3f(*color)  # yellowish
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

	def draw_robot(self):
		tf = self.robot.eta
		glPushMatrix()
		glTranslatef(*tf[:3])
		glRotatef(tf[5] * RAD2DEG, 0.0, 0.0, 1.0)
		glRotatef(tf[4] * RAD2DEG, 0.0, 1.0, 0.0)
		glRotatef(tf[3] * RAD2DEG, 1.0, 0.0, 0.0)
		draw_vbo_textured(self.robot_vbo, self.robot_vertex_count, self.robot_texture_id)
		if self.bool_draw_axis:
			self.draw_axis()
		glPopMatrix()

	def draw_cubic_robot(self, size=0.3):
		"""
		Draws a cubic AUV from multiple cubes to visualize orientation:
		- Main body cube
		- Forward cube (nose)
		- Top cube (sensor/antenna)
		- Left + Right cubes (thruster pods)
		"""

		def draw_cube(s):
			"""Draws a single cube centered at local origin."""
			h = s / 2.0
			verts = [
				[-h, -h, -h], [ h, -h, -h], [ h,  h, -h], [-h,  h, -h],  # Back
				[-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h],  # Front
			]
			faces = [
				(0, 1, 2, 3),
				(4, 5, 6, 7),
				(0, 1, 5, 4),
				(3, 2, 6, 7),
				(1, 2, 6, 5),
				(0, 3, 7, 4)
			]
			normals = [
				(0, 0, -1),
				(0, 0,  1),
				(0,-1,  0),
				(0, 1,  0),
				(1, 0,  0),
				(-1,0,  0)
			]
			glBegin(GL_QUADS)
			for face, normal in zip(faces, normals):
				glNormal3f(*normal)
				for idx in face:
					glVertex3f(*verts[idx])
			glEnd()


		tf = self.robot.eta
		glPushMatrix()
		glTranslatef(*tf[:3])
		glRotatef(tf[5] * RAD2DEG, 0.0, 0.0, 1.0)
		glRotatef(tf[4] * RAD2DEG, 0.0, 1.0, 0.0)
		glRotatef(tf[3] * RAD2DEG, 1.0, 0.0, 0.0)

		# --- MAIN BODY ---
		glPushMatrix()
		glColor3f(61/255, 209/255, 242/255)
		draw_cube(size)
		glPopMatrix()

		# --- NOSE CUBE (front) ---
		glPushMatrix()
		glColor3f(220/255, 225/255, 31/255)
		glTranslatef(size * 0.7, 0, 0)
		draw_cube(size * 0.4)
		glPopMatrix()

		# --- LEFT THRUSTER POD ---
		glPushMatrix()
		glColor3f(16/255, 79/255, 117/255)
		glTranslatef(0, -size * 0.55, 0)
		draw_cube(size * 0.2)
		glPopMatrix()

		# --- RIGHT THRUSTER POD ---
		glPushMatrix()
		glColor3f(16/255, 79/255, 117/255)
		glTranslatef(0, size * 0.55, 0)
		draw_cube(size * 0.2)
		glPopMatrix()

		if self.bool_draw_axis:
			self.draw_axis()


		tf = self.robot.cam_eta
		glTranslatef(*tf[:3])
		glRotatef(tf[5] * RAD2DEG, 0.0, 0.0, 1.0)
		glRotatef(tf[4] * RAD2DEG, 0.0, 1.0, 0.0)
		glRotatef(tf[3] * RAD2DEG, 1.0, 0.0, 0.0)


		if self.bool_draw_axis:
			self.draw_axis()
		glPopMatrix()



	""" ------------------------------------------------------ """
	""" --------------------   CONTROLS   -------------------- """
	""" ------------------------------------------------------ """

	def keyboard(self, key, x, y):
		if key == b' ':
			self.running = not self.running
		elif key in (b's', b'S'):
			self.running = False
			self.step_request = True
		elif key in (b'f', b'F'):
			self.bool_follow_robot = not self.bool_follow_robot
		elif key in (b'c', b'C'):
			self.bool_robot_view = not self.bool_robot_view
		elif key in (b'd', b'D'):
			self.bool_draw_wanted_markers = not self.bool_draw_wanted_markers
		elif key in (b'l', b'L'):
			self.camera_phi = 0.0001
			self.camera_theta = np.pi
		elif key == b'\x1b':  # ESC
			print("Exiting simulation...")
			glutLeaveMainLoop()

	def special_keys(self, key, x, y):
		if key == GLUT_KEY_LEFT:
			self.robot.dt = np.sign(self.robot.dt) * max(1/2**6, np.abs(self.robot.dt) / 2)
		elif key == GLUT_KEY_DOWN:
			self.robot.dt = 0.01
		elif key == GLUT_KEY_UP:
			self.robot.dt = 0.01
		elif key == GLUT_KEY_RIGHT:
			self.robot.dt = np.sign(self.robot.dt) * min(2**6, np.abs(self.robot.dt) * 2)
	
	def joystick_func(self, buttons, x, y, z):
		if not (x == y and y == z):
			if buttons == 2:
				self.robot.eta[2] += y * 50e-6   # for example, add x-axis
			else:
				self.robot.eta[0] += -y * 50e-6   # for example, add x-axis
			self.robot.eta[1] += x * 50e-6  # add y-axis
			# self.robot.eta[2] += z * 50e-6  # add z-axis
			if buttons == 2:
				self.robot.eta[4] += z * 50e-6   # for example, add x-axis
			else:
				self.robot.eta[5] += z * 50e-6  # add z-axis

	def mouse(self, button, state, x, y):
		if state == GLUT_DOWN:
			self.mouse_button = button
			self.mouse_prev = [x, y]
		else:
			self.mouse_button = None

		y = self.window_height - y  # Invert Y for UI coords

		if state != GLUT_DOWN:
			return
		
	def zoom(self, wheel, direction, x, y):
		self.camera_radius *= 0.9 if direction > 0 else 1.1
		self.camera_radius = np.clip(self.camera_radius, 1.0, 300.0)
		
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