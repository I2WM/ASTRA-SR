import numpy as np
import numpy.fft as fft
from scipy.ndimage import zoom, fourier_shift

from hcipy import (
    make_pupil_grid,
    make_focal_grid,
    FraunhoferPropagator,
    FresnelPropagator,
    circular_aperture,
    Wavefront,
    Field,
    InfiniteAtmosphericLayer,
)

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

def _make_focal_grid_compat(q, num_airy, pupil_diameter, wavelength_m):
    try:
        return make_focal_grid(q, num_airy, pupil_diameter=pupil_diameter, focal_length=1.0, reference_wavelength=wavelength_m)
    except TypeError:
        return make_focal_grid(q, num_airy, pupil_diameter=pupil_diameter, focal_length=1.0, wavelength=wavelength_m)

class SplitStepPSFSimulatorCPU:
    def __init__(self, cfg):
        self.cfg = cfg

    def _random_wind_dirs(self, n_layers: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 2.0 * np.pi, size=n_layers)

    def _crop_around_peak(self, psf2d: np.ndarray, out_size: int) -> np.ndarray:
        H, W = psf2d.shape
        k = out_size // 2
        py, px = np.unravel_index(np.argmax(psf2d), psf2d.shape)
        y0, x0 = py - k, px - k
        y1, x1 = y0 + out_size, x0 + out_size
        out = np.zeros((out_size, out_size), dtype=np.float32)
        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(H, y1), min(W, x1)
        dy0, dx0 = sy0 - y0, sx0 - x0
        dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
        out[dy0:dy1, dx0:dx1] = psf2d[sy0:sy1, sx0:sx1].astype(np.float32, copy=False)
        s = float(out.sum())
        if s > 0:
            out /= s
        return out

    def _recenter_kernel_centroid(self, kernel: np.ndarray) -> np.ndarray:
        K = kernel.shape[0]
        y, x = np.mgrid[0:K, 0:K]
        s = float(kernel.sum()) + 1e-12
        cy = float((kernel * y).sum() / s)
        cx = float((kernel * x).sum() / s)
        dy = (K - 1) / 2.0 - cy
        dx = (K - 1) / 2.0 - cx
        F = fft.fftn(kernel)
        shifted = np.real(fft.ifftn(fourier_shift(F, shift=(dy, dx))))
        shifted = np.maximum(shifted, 0.0)
        shifted /= (float(shifted.sum()) + 1e-12)
        return shifted.astype(np.float32)

    def _resample_psf_to_image_pixels(self, psf2d_focal: np.ndarray, D_m: float) -> np.ndarray:
        wavelength_m = float(self.cfg.optics.wavelength_m)
        focal_q = int(self.cfg.sampling.focal_q)
        plate_scale = float(self.cfg.sampling.plate_scale_arcsec_per_px)
        
        psf_pix_rad = (wavelength_m / D_m) / float(focal_q)
        img_pix_rad = plate_scale * ARCSEC_TO_RAD
        z = psf_pix_rad / img_pix_rad
        out = zoom(psf2d_focal, zoom=(z, z), order=1)
        out = np.maximum(out, 0.0)
        out /= (float(out.sum()) + 1e-12)
        return out.astype(np.float32)

    def _build_optics_and_layers(self, cn2_layers, wind_speeds):
        # Read from optics and sampling configs
        D_m = float(self.cfg.optics.D_m)
        wavelength_m = float(self.cfg.optics.wavelength_m)
        outer_scale = float(self.cfg.optics.outer_scale_m)
        pupil_samples = int(self.cfg.sampling.pupil_samples)
        pupil_pad = float(self.cfg.sampling.pupil_pad_factor)
        heights = np.array(self.cfg.atmosphere.heights_m, dtype=float)
        seed = int(self.cfg.simulation.seed)
        
        N = len(cn2_layers)
        wind_dirs = self._random_wind_dirs(N, seed=seed + 12345)
        
        pupil_grid = make_pupil_grid(pupil_samples, diameter=pupil_pad * D_m)
        coords = pupil_grid.as_("cartesian")
        aperture = circular_aperture(D_m)(pupil_grid)
        
        np.random.seed(seed)
        
        layers = []
        for i in range(N):
            v = float(wind_speeds[i])
            ang = float(wind_dirs[i])
            vx, vy = v * np.cos(ang), v * np.sin(ang)
            # Clip Cn2 to prevent division by zero in hcipy (matches elegant fix for Run 11)
            cn2_val = max(float(cn2_layers[i]), 1e-20)
            layer = InfiniteAtmosphericLayer(
                pupil_grid,
                cn2_val,
                outer_scale,
                (vx, vy),
                seed=seed + i
            )
            layer.height = float(heights[i])
            layers.append(layer)
            
        order = np.argsort(heights)[::-1]
        heights_sorted = heights[order]
        layers_sorted = [layers[i] for i in order]
        
        fresnels = []
        for k in range(len(heights_sorted) - 1):
            d = float(heights_sorted[k] - heights_sorted[k + 1])
            fresnels.append(
                FresnelPropagator(pupil_grid, d, num_oversampling=1, zero_padding=1)
                if d > 0 else None
            )
            
        d_ground = float(heights_sorted[-1])
        fresnel_to_ground = (
            FresnelPropagator(pupil_grid, d_ground, num_oversampling=1, zero_padding=1)
            if d_ground > 0 else None
        )
        
        focal_grid = _make_focal_grid_compat(
            int(self.cfg.sampling.focal_q),
            int(self.cfg.sampling.focal_num_airy),
            pupil_diameter=D_m,
            wavelength_m=wavelength_m
        )
        to_focal = FraunhoferPropagator(pupil_grid, focal_grid)
        
        meta = {
            "wavelength_m": wavelength_m,
            "outer_scale_m": outer_scale,
            "D_m": D_m,
            "image_size": int(self.cfg.sampling.image_size),
            "plate_scale_arcsec_per_px": float(self.cfg.sampling.plate_scale_arcsec_per_px),
            "sim_grid": int(self.cfg.sampling.sim_grid),
            "out_grid": int(self.cfg.sampling.out_grid),
            "kernel_size": int(self.cfg.sampling.kernel_size),
            "pupil_samples": pupil_samples,
            "pupil_pad_factor": pupil_pad,
            "focal_q": int(self.cfg.sampling.focal_q),
            "focal_num_airy": int(self.cfg.sampling.focal_num_airy),
            "heights_m": heights,
            "wind_dirs_rad": wind_dirs,
        }
        return pupil_grid, coords, aperture, layers_sorted, fresnels, fresnel_to_ground, to_focal, meta

    def run(self, cn2_layers, wind_speeds):
        M = int(self.cfg.simulation.M)
        delta_t = float(self.cfg.simulation.delta_t)
        
        pupil_grid, coords, aperture, layers_sorted, fresnels, fresnel_to_ground, to_focal, meta = \
            self._build_optics_and_layers(cn2_layers, wind_speeds)
            
        # FOV coverage
        fov_rad = (float(self.cfg.sampling.plate_scale_arcsec_per_px) * ARCSEC_TO_RAD) * int(self.cfg.sampling.image_size)
        theta_max = 0.5 * fov_rad
        sim_grid = int(self.cfg.sampling.sim_grid)
        idx = (np.arange(sim_grid) + 0.5) / sim_grid
        theta_1d = (idx - 0.5) * 2.0 * theta_max
        
        k0 = 2.0 * np.pi / float(self.cfg.optics.wavelength_m)
        D_m = float(self.cfg.optics.D_m)
        K = int(self.cfg.sampling.kernel_size)
        out_grid = int(self.cfg.sampling.out_grid)
        
        psf_sum = [[0.0 for _ in range(sim_grid)] for _ in range(sim_grid)]
        
        for m in range(M):
            t = float(m * delta_t)
            for layer in layers_sorted:
                layer.evolve_until(t)
                
            for iy in range(sim_grid):
                thy = float(theta_1d[iy])
                for ix in range(sim_grid):
                    thx = float(theta_1d[ix])
                    
                    ramp = np.exp(1j * k0 * (thx * coords.x + thy * coords.y))
                    wf = Wavefront(Field(ramp, pupil_grid), float(self.cfg.optics.wavelength_m))
                    
                    for k, layer in enumerate(layers_sorted):
                        wf.electric_field *= np.exp(1j * layer.phase_for(float(self.cfg.optics.wavelength_m)))
                        if k < len(fresnels) and fresnels[k] is not None:
                            wf = fresnels[k](wf)
                            
                    if fresnel_to_ground is not None:
                        wf = fresnel_to_ground(wf)
                        
                    wf.electric_field *= aperture
                    img = to_focal(wf).intensity.shaped
                    psf_sum[iy][ix] = psf_sum[iy][ix] + img
                    
        kernels_sim = np.zeros((sim_grid, sim_grid, K, K), dtype=np.float32)
        for iy in range(sim_grid):
            for ix in range(sim_grid):
                psf_le = (psf_sum[iy][ix] / float(M)).astype(np.float32, copy=False)
                psf_le = np.maximum(psf_le, 0.0)
                psf_le /= (float(psf_le.sum()) + 1e-12)
                
                psf_img = self._resample_psf_to_image_pixels(psf_le, D_m=D_m)
                ker = self._crop_around_peak(psf_img, K)
                ker = self._recenter_kernel_centroid(ker)
                kernels_sim[iy, ix] = ker
                
        zoom_y = out_grid / sim_grid
        kernels32 = zoom(kernels_sim, zoom=(zoom_y, zoom_y, 1.0, 1.0), order=1).astype(np.float32)
        s = kernels32.sum(axis=(2, 3), keepdims=True) + 1e-12
        kernels32 /= s
        
        meta = dict(meta)
        meta.update({"M": int(M), "delta_t": float(delta_t)})
        return kernels32, meta
