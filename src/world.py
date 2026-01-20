from numpy import array
from scipy.spatial.transform import Rotation as R
from ocean import Ocean
from cable import Cable

class World:

	def __init__(self, world_params):

		self.gravity = world_params['gravity']

		self.ocean = Ocean(world_params['ocean'])


		markers_params = world_params['markers']
		self.markers_init_pos = markers_params['initial_position']

		self.t_markers = markers_params['translation']
		rpy_markers = markers_params['rotation']
		self.R_markers = R.from_euler('xyz', rpy_markers).as_matrix()

		self.marker_colors = markers_params['colors']

		self.markers = array([self.t_markers + self.R_markers @ m for m in self.markers_init_pos])


		cables_params = world_params['cables']
		self.cables = []

		for cable in cables_params:
			self.cables.append(
				Cable(
					anchor1_pos=cable['initial_position'],
					anchor2_pos=cable['end_position'],
					length=cable['length'],
					radius=cable['radius'],
					n_subdiv=cable['n_subdiv'],
					linear_mass=cable['linear_mass'],
					bending_coef=cable['bending_coef'],
					jokobsen_params=cable['jakobsen_params'],
					anchors_fixed=cable['anchored'],
					color=cable['color'],
				)
			)


	def update(self, eta, dt, time):
		self.ocean.update(eta, dt, time)


		for cable in self.cables:
			cable.update(dt, self)
