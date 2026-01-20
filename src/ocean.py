import numpy as np

class Ocean:
    def __init__(self, environment_params):
        # Physical environment properties (required)
        self.water_density = environment_params["water_density"]
        self.water_viscosity = environment_params["water_viscosity"]

        # Current configuration (type is required)
        self.current_types = environment_params["current"]["types"].split(',')
        self.current_params = environment_params["current"]

        self.fluid_vel = np.zeros(6)
        self.time = 0

        # Preprocess data depending on current type
        if "time_series" in self.current_types:
            self.time_series = sorted(self.current_params["time_series"], key=lambda d: d["time"])

        elif "depth_profile" in self.current_types:
            self.depth_profile = sorted(self.current_params["depth_profile"], key=lambda d: d["depth"])

    def init_fluid_vel(self):
        for current_type in self.current_types:
            if current_type == "normal":
                mean = np.array(self.current_params["normal"]["speed"])
                std = np.array(self.current_params["normal"]["std"])
                self.fluid_vel = np.random.normal(mean, std)
            elif current_type == "constant":
                self.fluid_vel = np.array(self.current_params["constant"]["vector"])


    def update(self, pos, dt, time=None):
        """
        Compute the 6D fluid velocity vector at a given position.
        pos: [x, y, z] as numpy array
        time: optional time time for time_series currents
        """
        self.time += dt

        for current_type in self.current_types:
            if current_type == "zero":
                self.fluid_vel = np.zeros(6)

            elif current_type == "normal":
                mean = np.array(self.current_params["normal"]["speed"])
                std = np.array(self.current_params["normal"]["std"])
                self.fluid_vel += (mean - np.random.normal(mean, std)) * dt

            elif current_type == "constant":
                self.fluid_vel = np.array(self.current_params["constant"]["vector"])

            elif current_type == "time_series":
                if self.time <= self.time_series[0]["time"]:
                    self.fluid_vel = np.array(self.time_series[0]["vector"])
                    continue
                if self.time >= self.time_series[-1]["time"]:
                    self.fluid_vel = np.array(self.time_series[-1]["vector"])
                    continue

                for i in range(len(self.time_series) - 1):
                    d0 = self.time_series[i]["time"]
                    d1 = self.time_series[i + 1]["time"]
                    if d0 <= self.time <= d1:
                        v0 = np.array(self.time_series[i]["vector"])
                        v1 = np.array(self.time_series[i + 1]["vector"])
                        ratio = (self.time - d0) / (d1 - d0)
                        self.fluid_vel = v0 + ratio * (v1 - v0)


            elif current_type == "depth_profile":
                z = pos[2]
                if z <= self.depth_profile[0]["depth"]:
                    self.fluid_vel = np.array(self.depth_profile[0]["vector"])
                    continue
                if z >= self.depth_profile[-1]["depth"]:
                    self.fluid_vel = np.array(self.depth_profile[-1]["vector"])
                    continue

                for i in range(len(self.depth_profile) - 1):
                    d0 = self.depth_profile[i]["depth"]
                    d1 = self.depth_profile[i + 1]["depth"]
                    if d0 <= z <= d1:
                        v0 = np.array(self.depth_profile[i]["vector"])
                        v1 = np.array(self.depth_profile[i + 1]["vector"])
                        ratio = (z - d0) / (d1 - d0)
                        self.fluid_vel = v0 + ratio * (v1 - v0)

            elif current_type == "jet":
                jet_force =  np.array(self.current_params["jet"]["vector"])
                jet_period   = self.current_params["jet"]["period"]
                jet_duty     = self.current_params["jet"]["duty"]

                phase = (time % jet_period) / jet_period
                jet_vector = jet_force * (phase < jet_duty)

                self.fluid_vel += jet_vector

            else:
                raise ValueError(f"Unsupported current type: {current_type}.")
