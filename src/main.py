import time, os
import signal
from numpy import set_printoptions, inf
from viewer.renderer import Renderer
from data import Data

set_printoptions(threshold=inf, linewidth=inf, formatter={'float': lambda x: "{0:.3e}".format(x)})

def main():	# handle CTRL + C
	viewer = Renderer()
	data = Data()

	def handle_sigint(signum, frame):
		sim_manager.viewer.exit = True
	signal.signal(signal.SIGINT, handle_sigint)

	if viewer is not None:
		viewer.run(data)



if __name__ == '__main__':
	main()
