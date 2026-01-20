from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class Button:
	def __init__(self, x, y, width, height, label, onclick=None,
				 bg_color=(0.2, 0.2, 0.2), hover_color=(0.3, 0.3, 0.3),
				 click_color=(0.1, 0.4, 0.6), hover_click_color=(0.2, 0.5, 0.7), text_color=(1, 1, 1), active=False):

		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.label = label
		self.onclick = onclick

		self.bg_color = bg_color
		self.hover_color = hover_color
		self.hover_click_color = hover_click_color
		self.click_color = click_color
		self.text_color = text_color

		self.hovered = False
		self.pressed = False
		self.active = active  # For toggle state, optional

	def contains(self, mx, my):
		return self.x <= mx <= self.x + self.width and self.y <= my <= self.y + self.height

	def handle_mouse_motion(self, mx, my):
		self.hovered = self.contains(mx, my)

	def handle_mouse(self, state):
		if self.hovered:
			if state == GLUT_DOWN:
				self.active = not self.active
				if self.onclick:
					self.onclick()
		else:
			self.pressed = False
	
	def update_position(self, x, y, width, height):
		self.x = x
		self.y = y
		self.width = width
		self.heigth = height

	def draw(self):
		# Choose color based on state
		if self.active and self.hovered:
			color = self.hover_click_color
		elif self.active :
			color = self.click_color
		elif self.hovered:
			color = self.hover_color
		else:
			color = self.bg_color

		glColor3f(*color)
		glBegin(GL_QUADS)
		glVertex2f(self.x, self.y)
		glVertex2f(self.x + self.width, self.y)
		glVertex2f(self.x + self.width, self.y + self.height)
		glVertex2f(self.x, self.y + self.height)
		glEnd()

		# Draw label aligned to the right side (10px padding)
		text_width = len(self.label) * 9  # Approx width per char
		text_x = self.x + self.width + 10
		text_y = self.y + self.height // 2 - 5

		glColor3f(*self.text_color)
		glRasterPos2f(text_x, text_y)
		for ch in self.label:
			glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(ch))
