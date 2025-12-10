from viewer.viewer import Viewer
from robot import Robot

from numpy import set_printoptions, inf

def main():
	set_printoptions(threshold=inf, linewidth=inf, formatter={'float': lambda x: "{0:.3f}".format(x)})

	dt = 0.0001

	auv = Robot(None, dt)
	
	viewer = Viewer(dt)
	viewer.run(auv)


if __name__ == "__main__":
	main()
