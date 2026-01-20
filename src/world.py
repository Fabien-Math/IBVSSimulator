from numpy import array
from scipy.spatial.transform import Rotation as R

class World:
    def __init__(self, world_params):
        self.markers_init_pos = world_params['initial_position']

        self.t_markers = world_params['translation']
        rpy_markers = world_params['rotation']
        self.R_markers = R.from_euler('xyz', rpy_markers).as_matrix()

        self.marker_colors = world_params['colors']

        self.markers = array([self.t_markers + self.R_markers @ m for m in self.markers_init_pos])
