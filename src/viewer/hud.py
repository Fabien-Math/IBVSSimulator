import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

RAD2DEG = 180 / 3.1415926535
DEG2RAD = 3.1415926535 / 180
class HUD:
	def __init__(self, robot, window_width, window_height):
		self.robot = robot

		self.window_width = window_width
		self.window_height = window_height
		self.ratio = self.window_width / self.window_height


	def update_window_size(self, window_width, window_height):
		self.window_width = window_width
		self.window_height = window_height
		self.ratio = self.window_width / self.window_height


	def draw(self):
		self.eta = self.robot.eta
		self.nu = self.robot.nu

		self.draw_robot_info_hud()
		self.draw_current_hud()
		# self.draw_ladders()
		self.draw_heading_ladder()
		self.draw_pitch_ladder()


	def draw_current_hud(self, radius=60):
		"""
		Draw a round HUD panel in the bottom-right showing the current vector.
		"""

		# ---- Retrieve current vector ----
		current = self.robot.fluid_vel[:3].copy()
		mag = np.linalg.norm(current)
		if mag > 1e-8:
			direction = current / mag
		else:
			direction = np.array([0, 0, 0], dtype=float)

		# ---- Screen size ----
		viewport = glGetIntegerv(GL_VIEWPORT)
		screen_w, screen_h = viewport[2], viewport[3]

		# ---- Switch to 2D mode ----
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		glOrtho(0, screen_w, 0, screen_h, -1, 1)

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

		# ---- Panel center ----
		cx = screen_w - radius - 20
		cy = radius + 20

		# ------------------------------------------
		# Draw circular background
		# ------------------------------------------
		glColor4f(0.08, 0.08, 0.08, 0.7)
		glBegin(GL_TRIANGLE_FAN)
		glVertex2f(cx, cy)
		for i in range(0, 361, 3):
			ang = np.radians(i)
			glVertex2f(cx + np.cos(ang) * radius,
					cy + np.sin(ang) * radius)
		glEnd()

		# ------------------------------------------
		# Draw outline
		# ------------------------------------------
		glColor3f(0.25, 0.25, 0.25)
		glLineWidth(2)
		glBegin(GL_LINE_LOOP)
		for i in range(0, 361, 3):
			ang = np.radians(i)
			glVertex2f(cx + np.cos(ang) * radius,
					cy + np.sin(ang) * radius)
		glEnd()
		glLineWidth(1)

		# ------------------------------------------
		# Draw direction vector (arrow)
		# ------------------------------------------
		vec_length = radius * 0.75  # stay inside circle

		glColor3f(0.3, 0.6, 1.0)
		glLineWidth(2)
		glBegin(GL_LINES)
		glVertex2f(cx, cy)
		glVertex2f(cx + direction[0] * vec_length,
				cy + direction[1] * vec_length)
		glEnd()

		# ---- Draw arrow head ----
		if mag > 1e-8:
			end_x = cx + direction[0] * vec_length
			end_y = cy + direction[1] * vec_length

			left = np.array([-direction[1], direction[0]])
			right = -left

			ah = 10  # arrowhead size
			glBegin(GL_TRIANGLES)
			glVertex2f(end_x, end_y)
			glVertex2f(end_x + left[0] * ah,
					end_y + left[1] * ah)
			glVertex2f(end_x + right[0] * ah,
					end_y + right[1] * ah)
			glEnd()

		glLineWidth(1)

		# ------------------------------------------
		# Draw magnitude text under the circle
		# ------------------------------------------
		text = f"{mag:.2f}"

		glColor3f(1, 1, 1)
		glRasterPos2f(cx - 10, cy - radius - 15)

		for ch in text:
			glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(ch))

		# ---- Restore state ----
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_LIGHTING)

		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)
		glPopMatrix()

	def draw_heading_cylinder(self):
		"""
		Draw a rolling heading scale curved like a cylinder.
		heading: degrees [0..360)
		"""

		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		gluOrtho2D(0,1,0,1)

		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()
		glDisable(GL_DEPTH_TEST)

		heading = self.eta[5] * RAD2DEG

		# Tape parameters
		tick_step = 5             # tick every 10°
		label_step = 15            # label every 30°

		glColor3f(0.05, 0.05, 0.05)
		glLineWidth(2)

		glPushMatrix()

		# CYLINDER PROJECTION PARAMETERS
		# CYLINDER CENTERED AT (0,0)
		cx = 0.0          # fixed center X
		cy = 0.0          # fixed center Y
		R  = 0.075         # radius of circle / cylinder

		for deg in range(-180 + tick_step, 181, tick_step):

			# Convert tape degree to arc angle around circle
			theta = (deg - heading + 45) * DEG2RAD

			tick_len  = 0.01
			label_len = 0.017

			# POSITION ON THE CIRCLE
			x = cx + R * np.sin(theta)
			y = cy + R * self.ratio * np.cos(theta)
			xt = cx + (R + tick_len) * np.sin(theta)
			yt = cy + (R + tick_len) * self.ratio * np.cos(theta)
			xl = cx + (R + label_len) * np.sin(theta)
			yl = cy + (R + label_len) * self.ratio * np.cos(theta)

			# ---- TICK ----
			glBegin(GL_LINES)
			glVertex2f(x, y)
			glVertex2f(xt, yt)
			glEnd()

			# ---- LABELS ----
			if deg % label_step == 0:
				glBegin(GL_LINES)
				glVertex2f(x, y)
				glVertex2f(xl, yl)
				glEnd()

				# label position slightly outside circle
				lx = cx + (R - 0.015) * np.sin(theta)
				ly = cy + (R - 0.015) * np.cos(theta) * self.ratio

				self.draw_text(lx, ly, f"{deg:g}", font=GLUT_BITMAP_8_BY_13, color=(0.05, 0.05, 0.05))


		glPopMatrix()

		offset = 0.005
		# Draw center marker
		glColor3f(0.3, 0.8, 1.0)
		glBegin(GL_TRIANGLES)
		glVertex2f(cx + R - offset, (cy + R - offset) * self.ratio)
		glVertex2f(cx + R - offset + 0.005, (cy + R - offset + 0.015) * self.ratio)
		glVertex2f(cx + R - offset + 0.015, (cy + R - offset + 0.005) * self.ratio)
		glEnd()

		self.draw_text(R + 1.5 * offset, (R + 1.5 * offset) * self.ratio, f"{int(heading):g}°", font=GLUT_BITMAP_8_BY_13, color=(0.05, 0.05, 0.05))


		glLineWidth(1)

		# ---- Restore state ----
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_LIGHTING)

		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)
		glPopMatrix()

	def draw_heading_ladder(self):
		"""
		Draw a rolling heading scale curved like a cylinder.
		heading: degrees [0..360)
		"""

		if np.isnan(self.eta[5]):
			return

		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		gluOrtho2D(0,1,0,1)

		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()
		glDisable(GL_DEPTH_TEST)


		heading = self.eta[5] * RAD2DEG % 360
		heading = heading * (abs(heading) <= 180) + (heading - 360) * (heading > 180) + (heading + 360) * (heading < -180)


		# Tape parameters
		tick_step = 5             # tick every 10°
		label_step = 15            # label every 30°

		glColor3f(0.05, 0.05, 0.05)
		glLineWidth(2)

		glPushMatrix()

		# CYLINDER PROJECTION PARAMETERS
		# CYLINDER CENTERED AT (0,0)
		cx = 0.16          # fixed center X
		cy = 0.025          # fixed center Y
		cyl = 0.01
		width = 0.3
		angle_span = 120

		for deg in range(int(-180 + tick_step - angle_span / 2), int(181 + angle_span / 2), tick_step):

			r = (deg - heading) / 180
			if abs(r) > angle_span / 2 / 180:
				continue

			r *= width

			tick_len  = 0.01
			label_len = 0.017

			# POSITION ON THE CIRCLE
			x = cx + r
			y = cy
			xt = cx + r
			yt = cy + tick_len
			xl = cx + r
			yl = cy + label_len

			# ---- TICK ----
			glBegin(GL_LINES)
			glVertex2f(x, y)
			glVertex2f(xt, yt)
			glEnd()

			# ---- LABELS ----
			if deg % label_step == 0:
				glBegin(GL_LINES)
				glVertex2f(x, y)
				glVertex2f(xl, yl)
				glEnd()

				# label position slightly outside circle
				lx = cx + r
				ly = cyl
				deg_label = f"{deg * (abs(deg) <= 180) + (deg - 360) * (deg > 180) + (deg + 360) * (deg < -180):g}"
				label_size = 8 / self.window_width * len(deg_label)
				self.draw_text(lx - label_size / 2, ly, deg_label, font=GLUT_BITMAP_8_BY_13, color=(0.05, 0.05, 0.05))


		glPopMatrix()

		offset = 0.005
		# Draw center marker
		glColor3f(0.3, 0.8, 1.0)
		glBegin(GL_TRIANGLES)
		glVertex2f(cx, cy + 0.02)
		glVertex2f(cx - 0.005, cy + 0.02 + 0.015)
		glVertex2f(cx + 0.005, cy + 0.02 + 0.015)
		glEnd()
		heading_label = f"{int(heading):g}°"
		label_size = 8 / self.window_width * (len(heading_label) - 1)
		self.draw_text(cx - label_size / 2, cy + 0.02 + 0.02, heading_label, font=GLUT_BITMAP_8_BY_13, color=(0.05, 0.05, 0.05))


		glLineWidth(1)

		# ---- Restore state ----
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_LIGHTING)

		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)
		glPopMatrix()

	def draw_pitch_ladder(self):
		"""
		Draw a rolling heading scale curved like a cylinder.
		heading: degrees [0..360)
		"""
		if np.isnan(self.eta[4]):
			return

		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		gluOrtho2D(0,1,0,1)

		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()
		glDisable(GL_DEPTH_TEST)


		pitch = self.eta[4] * RAD2DEG
		pitch = pitch * (abs(pitch) <= 90) + (pitch - 180) * (pitch > 90) + (pitch + 180) * (pitch < -90)


		# Tape parameters
		tick_step = 5             # tick every 10°
		label_step = 15            # label every 30°

		glColor3f(0.05, 0.05, 0.05)
		glLineWidth(2)

		glPushMatrix()

		# CYLINDER PROJECTION PARAMETERS
		# CYLINDER CENTERED AT (0,0)
		cx = 0.025          # fixed center X
		cy = 0.20        # fixed center Y
		cxl = 0.01
		height = 0.3
		angle_span = 90
		tick_len  = 0.01 / self.ratio
		label_len = 0.017 / self.ratio

		for deg in range(int(-90 + tick_step - angle_span / 2), int(90 + angle_span / 2), tick_step):

			r = (deg - pitch) / 180
			if abs(r) > angle_span / 2 / 180:
				continue

			r *= height


			# POSITION ON THE CIRCLE
			x = cx
			y = cy + r * self.ratio
			xt = cx + tick_len
			yt = cy + r * self.ratio
			xl = cx + label_len
			yl = cy + r * self.ratio

			# ---- TICK ----
			glBegin(GL_LINES)
			glVertex2f(x, y)
			glVertex2f(xt, yt)
			glEnd()

			# ---- LABELS ----
			if deg % label_step == 0:
				glBegin(GL_LINES)
				glVertex2f(x, y)
				glVertex2f(xl, yl)
				glEnd()

				# label position slightly outside circle
				lx = cxl
				ly = cy + r * self.ratio
				deg_label = f"{deg * (abs(deg) <= 90) + (deg - 180) * (deg > 90) + (deg + 180) * (deg < -90):g}"
				label_size = 8 / self.window_width * len(deg_label)
				self.draw_text(lx, ly - label_size / 2, deg_label, font=GLUT_BITMAP_8_BY_13, color=(0.05, 0.05, 0.05))


		glPopMatrix()

		offset = 0.005
		# Draw center marker
		glColor3f(0.3, 0.8, 1.0)
		glBegin(GL_TRIANGLES)
		glVertex2f(cx + 0.015, cy)
		glVertex2f(cx + 0.015 + 0.01, cy - 0.005 * self.ratio)
		glVertex2f(cx + 0.015 + 0.01, cy + 0.005 * self.ratio)
		glEnd()
		pitch_label = f"{int(pitch):g}°"
		label_size = 8 / self.window_width * (len(pitch_label) - 1)
		self.draw_text(cx + 0.015 + 0.015, cy - label_size / 2, pitch_label, font=GLUT_BITMAP_8_BY_13, color=(0.05, 0.05, 0.05))


		glLineWidth(1)

		# ---- Restore state ----
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_LIGHTING)

		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)
		glPopMatrix()

	def draw_text(self, x, y, text, font=GLUT_BITMAP_9_BY_15, color=(1, 1, 1)):
		glColor3f(*color)
		glRasterPos2f(x, y)
		for ch in text:
			glutBitmapCharacter(font, ord(ch))

	def draw_robot_info_hud(self, line_height=20, panel_width=300, panel_padding=15, border_thickness=2):
		"""
		Draw a cockpit-style HUD for the robot.
		Uses self.eta and self.nu.
		"""

		# --- Save OpenGL state ---
		glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_LINE_BIT)
		glDisable(GL_DEPTH_TEST)
		glDisable(GL_LIGHTING)

		# --- Save matrices ---
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()

		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()

		# --- Setup orthographic projection ---
		viewport = glGetIntegerv(GL_VIEWPORT)
		width, height = viewport[2], viewport[3]
		glOrtho(0, width, 0, height, -1, 1)

		# --- Enable blending / line smoothing ---
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable(GL_LINE_SMOOTH)
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

		# --- Flip Y-axis so (0,0) is top-left ---
		glTranslatef(0, height, 0)
		glScalef(1, -1, 1)

		# --- Panel position: center-bottom ---
		panel_height = line_height + 2 * panel_padding
		panel_x = (width - panel_width) / 2
		panel_y = height - panel_height - 30  # margin from bottom

		# --- Draw rounded rectangle panel (approx) ---
		# For simplicity, use multiple small line segments for rounded corners
		corner_radius = 10
		segments = 16

		def draw_rounded_rect(x, y, w, h, r):
			glBegin(GL_POLYGON)
			# Bottom-left corner
			for i in range(segments + 1):
				theta = np.pi + (np.pi / 2) * (i / segments)
				glVertex2f(x + r + r * np.cos(theta), y + r + r * np.sin(theta))
			# Bottom-right
			for i in range(segments + 1):
				theta = 1.5 * np.pi + (np.pi / 2) * (i / segments)
				glVertex2f(x + w - r + r * np.cos(theta), y + r + r * np.sin(theta))
			# Top-right
			for i in range(segments + 1):
				theta = 0 + (np.pi / 2) * (i / segments)
				glVertex2f(x + w - r + r * np.cos(theta), y + h - r + r * np.sin(theta))
			# Top-left
			for i in range(segments + 1):
				theta = 0.5 * np.pi + (np.pi / 2) * (i / segments)
				glVertex2f(x + r + r * np.cos(theta), y + h - r + r * np.sin(theta))
			glEnd()

		# Panel background
		# glColor4f(0.08, 0.08, 0.08, 0.7)
		glColor3f(0.8, 0.8, 0.8)
		glLineWidth(border_thickness)
		draw_rounded_rect(panel_x, panel_y, panel_width, panel_height, corner_radius)
		glLineWidth(1)

		# Panel border
		glColor4f(0.08, 0.08, 0.08, 0.7)
		draw_rounded_rect(panel_x, panel_y, panel_width, panel_height, corner_radius)


		# --- Draw speed bar ---
		speed = np.linalg.norm(self.nu[:3])
		max_speed = 1.5  # adjust as needed
		bar_width = panel_width - 2 * panel_padding
		bar_height = 10
		bar_x = panel_x + panel_padding
		bar_y = panel_y + panel_padding

		# Background
		glColor3f(0.2, 0.2, 0.2)
		glBegin(GL_QUADS)
		glVertex2f(bar_x, bar_y)
		glVertex2f(bar_x + bar_width, bar_y)
		glVertex2f(bar_x + bar_width, bar_y + bar_height)
		glVertex2f(bar_x, bar_y + bar_height)
		glEnd()

		# Filled speed
		filled_width = min(bar_width, bar_width * (speed / max_speed))
		if speed > max_speed:
			glColor3f(0.9, 0.3, 0.2)
		else:
			glColor3f(0.3, 0.8, 1.0)

		glBegin(GL_QUADS)
		glVertex2f(bar_x, bar_y)
		glVertex2f(bar_x + filled_width, bar_y)
		glVertex2f(bar_x + filled_width, bar_y + bar_height)
		glVertex2f(bar_x, bar_y + bar_height)
		glEnd()

		# --- Draw speed text ---
		self.draw_text(bar_x + bar_width / 2 - 20, bar_y + line_height * 1.5, f"{speed:.2f} m/s", color=(0.9, 0.9, 0.9))

		# --- Restore matrices and state ---
		glMatrixMode(GL_MODELVIEW)
		glPopMatrix()
		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glPopAttrib()
