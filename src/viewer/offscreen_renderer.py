from OpenGL.GL import *
import numpy as np

class OffscreenRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Create FBO
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # Create texture for color attachment
        self.color_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_tex, 0)

        # Create depth buffer
        self.depth_rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_rb)

        # Check FBO completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer not complete!")

        # Unbind FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def read_pixels(self):
        """Read pixels from the FBO into a NumPy array (BGR format)."""
        raw = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        img = np.flipud(img)[:, :, ::-1]  # RGB -> BGR
        return img

    def cleanup(self):
        glDeleteFramebuffers(1, [self.fbo])
        glDeleteTextures(1, [self.color_tex])
        glDeleteRenderbuffers(1, [self.depth_rb])
