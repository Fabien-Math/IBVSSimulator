import numpy as np
import ctypes
from OpenGL.GL import *
from OpenGL.GLUT import *

class GPUCable:
	def __init__(self, cable, segments=8):
		self.cable = cable
		self.segments = segments
		self.vertices = None
		self.normals = None
		self.indices = None
		self.vbo = None
		self.ebo = None
		self.vao = None
		self.n_indices = 0

		self.build_mesh()
		self.upload_mesh()

	def build_mesh(self):
		"""Build vertices, normals, and indices for a tube along cable points"""
		positions = self.cable.positions
		n_points = len(positions)
		if n_points < 2:
			self.vertices = np.zeros((0,3), dtype=np.float32)
			self.normals = np.zeros((0,3), dtype=np.float32)
			self.indices = np.zeros((0,), dtype=np.uint32)
			self.n_indices = 0
			return

		vertices = []
		normals = []
		indices = []

		for i in range(n_points):
			p = positions[i]

			# Tangent
			if i == 0:
				tangent = positions[i+1] - p
			elif i == n_points - 1:
				tangent = p - positions[i-1]
			else:
				tangent = positions[i+1] - positions[i-1]

			tangent /= np.linalg.norm(tangent)

			# Orthonormal frame
			up = np.array([0, 0, 1], dtype=np.float32)
			if abs(np.dot(up, tangent)) > 0.99:
				up = np.array([0, 1, 0], dtype=np.float32)

			x = np.cross(up, tangent)
			x /= np.linalg.norm(x)
			y = np.cross(tangent, x)

			for j in range(self.segments):
				theta = 2 * np.pi * j / self.segments
				offset = self.cable.radius * (np.cos(theta) * x + np.sin(theta) * y)
				vertices.append(p + offset)
				normals.append(offset / self.cable.radius)

		# Convert to arrays
		vertices = np.array(vertices, dtype=np.float32)
		normals = np.array(normals, dtype=np.float32)

		# Build triangle indices
		for i in range(n_points - 1):
			for j in range(self.segments):
				curr = i * self.segments + j
				next_seg = i * self.segments + (j + 1) % self.segments
				curr_next = (i + 1) * self.segments + j
				next_next = (i + 1) * self.segments + (j + 1) % self.segments

				indices.extend([curr, curr_next, next_next])
				indices.extend([curr, next_next, next_seg])

		indices = np.array(indices, dtype=np.uint32)

		self.vertices = vertices
		self.normals = normals
		self.indices = indices
		self.n_indices = len(indices)

	def upload_mesh(self):
		"""Upload vertices, normals, and indices to GPU"""
		self.vao = glGenVertexArrays(1)
		glBindVertexArray(self.vao)

		# VBO for positions and normals
		self.vbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
		vertex_data = np.hstack((self.vertices, self.normals)).astype(np.float32)
		glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)

		# Vertex positions
		glEnableClientState(GL_VERTEX_ARRAY)
		glVertexPointer(3, GL_FLOAT, 6*4, ctypes.c_void_p(0))

		# Normals
		glEnableClientState(GL_NORMAL_ARRAY)
		glNormalPointer(GL_FLOAT, 6*4, ctypes.c_void_p(12))

		# EBO for indices
		self.ebo = glGenBuffers(1)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_DYNAMIC_DRAW)

		# Unbind
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		glBindVertexArray(0)

	def update_positions(self):
		self.cable.update_positions()
		self.build_mesh()

		# Update GPU buffers
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
		vertex_data = np.hstack((self.vertices, self.normals)).astype(np.float32)
		glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)
		glBindBuffer(GL_ARRAY_BUFFER, 0)

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
		glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, self.indices.nbytes, self.indices)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

	def draw(self):
		"""Draw the cable tube with lighting"""
		glColor3f(*self.cable.color)
		glBindVertexArray(self.vao)
		glDrawElements(GL_TRIANGLES, self.n_indices, GL_UNSIGNED_INT, None)
		glBindVertexArray(0)

		# Draw end spheres for anchors
		glPushMatrix()
		glTranslatef(*self.cable.positions[0])
		glColor3f(0.1, 0.1, 0.1)
		glutSolidSphere(self.cable.radius*10, self.segments, self.segments)
		glPopMatrix()

		glPushMatrix()
		glTranslatef(*self.cable.positions[-1])
		glColor3f(1.0, 0.2, 0.1)
		glutSolidSphere(self.cable.radius*5, self.segments, self.segments)
		glPopMatrix()


# Complete CPU version to put inside rendered.py
# def draw_cables_cpu(self, segments=6):
# 	"""
# 	Draw cables as a mesh (tube) along positions using CPU only.
# 	segments: number of radial subdivisions around the cable
# 	"""

# 	for cable in self.world.cables:
# 		glColor3f(*cable.color)  # Orange-red cable
# 		cable.update_positions()
# 		positions = cable.positions
# 		n = len(positions)
# 		if n < 2:
# 			continue

# 		for i in range(n - 1):
# 			p0 = positions[i]
# 			p1 = positions[i + 1]
# 			dir_vec = p1 - p0
# 			length = np.linalg.norm(dir_vec)
# 			if length < 1e-6:
# 				continue

# 			# Local frame for cross-section
# 			z = dir_vec / length
# 			up = np.array([0, 0, 1])
# 			if np.abs(np.dot(up, z)) > 0.99:
# 				up = np.array([0, 1, 0])
# 			x = np.cross(up, z)
# 			x /= np.linalg.norm(x)
# 			y = np.cross(z, x)

# 			# Draw cylinder segment
# 			glBegin(GL_QUAD_STRIP)
# 			for j in range(segments + 1):
# 				theta = 2 * np.pi * j / segments
# 				offset = np.cos(theta) * x + np.sin(theta) * y
# 				normal = offset / np.linalg.norm(offset)  # normalize
# 				offset_scaled = offset * cable.radius

# 				# Vertex at start
# 				glNormal3f(*normal)
# 				glVertex3f(*(p0 + offset_scaled))

# 				# Vertex at end
# 				glNormal3f(*normal)
# 				glVertex3f(*(p1 + offset_scaled))
# 			glEnd()

# 		# Draw small spheres at anchors for clarity
# 		glPushMatrix()
# 		glTranslatef(*cable.positions[0])
# 		glColor3f(0.1, 0.1, 0.1)  # Orange-red cable
# 		glutSolidSphere(cable.radius*10, segments, segments)
# 		glPopMatrix()

# 		glPushMatrix()
# 		glTranslatef(*cable.positions[-1])
# 		glColor3f(1.0, 0.2, 0.1)  # Orange-red cable
# 		glutSolidSphere(cable.radius*5, segments, segments)
# 		glPopMatrix()
