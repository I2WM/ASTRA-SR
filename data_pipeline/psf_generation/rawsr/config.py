import yaml

class SimulationConfig:
    def __init__(self, d):
        self.seed = int(d.get("seed", 0))
        self.M = int(d.get("M", 8))
        self.delta_t = float(d.get("delta_t", 0.5))
        self.out_dir = str(d.get("out_dir", "outputs"))
        self.save_prefix = str(d.get("save_prefix", "new"))

class AtmosphereConfig:
    def __init__(self, d):
        self.csv_path = str(d.get("csv_path", ""))
        self.row_index = int(d.get("row_index", 0))
        self.n_layers = int(d.get("n_layers", 6))
        self.heights_m = [float(x) for x in d.get("heights_m", [500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0])]
        self.wind_speeds_mps = [float(x) for x in d.get("wind_speeds_mps", [])]

class OpticsConfig:
    def __init__(self, d):
        self.D_m = float(d.get("D_m", 0.5))
        self.wavelength_m = float(d.get("wavelength_m", 500e-9))
        self.outer_scale_m = float(d.get("outer_scale_m", 20.0))

class SamplingConfig:
    def __init__(self, d):
        self.image_size = int(d.get("image_size", 256))
        self.plate_scale_arcsec_per_px = float(d.get("plate_scale_arcsec_per_px", 0.1))
        self.out_grid = int(d.get("out_grid", 32))
        self.sim_grid = int(d.get("sim_grid", 16))
        self.kernel_size = int(d.get("kernel_size", 33))
        self.pupil_samples = int(d.get("pupil_samples", 128))
        self.pupil_pad_factor = float(d.get("pupil_pad_factor", 1.5))
        self.focal_q = int(d.get("focal_q", 2))
        self.focal_num_airy = int(d.get("focal_num_airy", 64))

class SimConfig:
    def __init__(self, raw):
        self.simulation = SimulationConfig(raw.get("simulation", {}))
        self.atmosphere = AtmosphereConfig(raw.get("atmosphere", {}))
        self.optics = OpticsConfig(raw.get("optics", {}))
        self.sampling = SamplingConfig(raw.get("sampling", {}))

def load_config(path: str) -> SimConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return SimConfig(raw)
