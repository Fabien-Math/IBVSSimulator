from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np



### WITH TEXTURE

def load_texture(texture_filename):
	image = Image.open(texture_filename).convert("RGB")
	image_data = image.tobytes("raw", "RGB", 0, -1)
	width, height = image.size

	tex_id = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, tex_id)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
				 GL_RGB, GL_UNSIGNED_BYTE, image_data)

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

	return tex_id


def load_obj_with_tex(obj_filename, texture_filename):
	vertices = []
	normals = []
	texcoords = []
	faces = []

	with open(obj_filename, 'r') as f:
		for line in f:
			if line.startswith('v '):
				parts = line.strip().split()
				vertices.append(list(map(float, parts[1:4])))
			elif line.startswith('vt '):
				parts = line.strip().split()
				texcoords.append(list(map(float, parts[1:3])))
			elif line.startswith('vn '):
				parts = line.strip().split()
				normals.append(list(map(float, parts[1:4])))
			elif line.startswith('f '):
				parts = line.strip().split()[1:]
				face = []
				for part in parts:
					v, vt, vn = (part.split('/') + [None, None])[:3]
					face.append((
						int(v) - 1 if v else None,
						int(vt) - 1 if vt else None,
						int(vn) - 1 if vn else None
					))
				faces.append(face)

	texture_id = load_texture(texture_filename)

	return vertices, texcoords, normals, faces, texture_id


def draw_obj_with_tex(vertices, normals, faces, texture_id, texcoords):
	glBindTexture(GL_TEXTURE_2D, texture_id)
	glEnable(GL_TEXTURE_2D)
	glBegin(GL_TRIANGLES)
	for face in faces:
		for v_idx, vt_idx, vn_idx in face:
			if vn_idx is not None:
				glNormal3fv(normals[vn_idx])
			if vt_idx is not None:
				glTexCoord2fv(texcoords[vt_idx])
			glVertex3fv(vertices[v_idx])
	glEnd()
	glDisable(GL_TEXTURE_2D)



### NO TEXTURE
def load_obj(filename):
	vertices = []
	normals = []
	faces = []

	with open(filename, 'r') as f:
		for line in f:
			if line.startswith('v '):
				parts = line.strip().split()
				vertex = list(map(float, parts[1:4]))
				vertices.append(vertex)
			elif line.startswith('vn '):
				parts = line.strip().split()
				normal = list(map(float, parts[1:4]))
				normals.append(normal)
			elif line.startswith('f '):
				parts = line.strip().split()[1:]
				face = []
				for part in parts:
					vals = part.split('//') if '//' in part else part.split('/')
					v_idx = int(vals[0]) - 1
					n_idx = int(vals[-1]) - 1 if len(vals) > 1 and vals[-1] != '' else None
					face.append((v_idx, n_idx))
				faces.append(face)
	
	return vertices, normals, faces

def draw_obj(vertices, normals, faces):
	glBegin(GL_TRIANGLES)
	for face in faces:
		for v_idx, n_idx in face:
			if n_idx is not None:
				glNormal3fv(normals[n_idx])
			glVertex3fv(vertices[v_idx])
	glEnd()


def create_vertex_data(vertices, texcoords, normals, faces):
    data = []
    for face in faces:
        for v_idx, vt_idx, vn_idx in face:
            # Position
            data.extend(vertices[v_idx])
            # Texture coords (fallback if missing)
            if texcoords and vt_idx is not None:
                data.extend(texcoords[vt_idx])
            else:
                data.extend([0.0, 0.0])
            # Normals (fallback if missing)
            if normals and vn_idx is not None:
                data.extend(normals[vn_idx])
            else:
                data.extend([0.0, 0.0, 0.0])
    return np.array(data, dtype=np.float32)

def create_vbo(data):
    vbo_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vbo_id


def draw_vbo_textured(vbo_id, vertex_count, texture_id):
    stride = 8 * 4  # 8 floats per vertex: 3 pos + 2 tex + 3 normal, 4 bytes each

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)

    # Set pointers to position, texture, normal in VBO
    glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
    glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(12))  # 3 floats * 4 bytes = 12
    glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(20))       # (3+2) floats * 4 bytes = 20

    glDrawArrays(GL_TRIANGLES, 0, vertex_count)

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
