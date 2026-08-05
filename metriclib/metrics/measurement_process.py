import numpy as np
import torch
import antropy as ant
import SimpleITK as sitk
from scipy.ndimage import binary_erosion
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from ..metric import MetricResult, StreamMetric, TabularMetric

import ast

class LimitofQuantification(StreamMetric):
    def aggregate(self, datapoint, reference=None, metric_config=None):
        if metric_config["cp"] is None:
            cp = 10
        else:
            cp = metric_config["cp"]

        if metric_config["LoB"] is None:
            raise ValueError("metric_config must include 'LoB' key")
        else:
            LoB = metric_config["LoB"]

        LoQ = LoB + cp * datapoint[0].mean()

        return LoQ

    def compute(self, data, reference, metric_config):
        return MetricResult(
            cluster=None,
            threshold=0,
            description="Proportion of values below limit of quantification",
            value=np.array(data).mean(),
        )

class SampleEntropy(StreamMetric):
    def aggregate(self, datapoint, reference=None, metric_config=None):
        metric_config = metric_config or {}
        max_points = int(metric_config.get("max_points_per_lead", 500))

        values = []
        for i in range(datapoint[0].shape[0]):
            x = np.asarray(datapoint[0][i], dtype=np.float64).ravel()
            if max_points > 1 and x.size > max_points:
                step = int(np.ceil(x.size / max_points))
                x = x[::step]
            x = np.ascontiguousarray(x)
            m = int(2)
            r = float(0.2 * np.std(x))
            values.append(
                ant.sample_entropy(x, order=m, tolerance=r, metric="chebyshev")
            )

        entropy = float(np.mean(values))

        return entropy

    def compute(self, data, reference, metric_config):
        return MetricResult(
            cluster="MeasurementProcess",
            threshold=0,
            description="Mean sample entropy across all leads",
            value=np.array(data).mean(),
        )

class SNR(StreamMetric):
    def aggregate(self, datapoint, reference=None, metric_config=None):
        if reference is None:
            raise ValueError("Reference signal is required for SNR calculation.")

        signal_power = np.mean(datapoint[0] ** 2)
        noise_power = np.mean((datapoint[0] - reference[0]) ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def compute(self, data, reference, metric_config):
        return MetricResult(
            cluster=None,
            threshold=0.001,
            description="Mean SNR across all leads",
            value=np.array(data).mean(),
        )

class MetadataCompleteness(TabularMetric):
    def compute(self, data, reference=None, metric_config=None):
        # count missing values in relation to all cells in the DataFrame
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)

        return MetricResult(
            description="Metadata Completeness",
            value=completeness,
            cluster="Measurement Process",
            threshold=1.0,
        )

class ImageEntropy3D(StreamMetric):
    """
    Computes Entropy of a CT Scan image (NIFTI format).
    """
    
    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requirement : 
        - source image in a tensor (datapoint[0])
        Optional :
        - number of bins for entropy approximation (in metric_config["bins"])(default : 256)
        
        Raises :
            ValueError : if "datapoint[0]" is not a torch.tensor
        Args:
            datapoint : 
                datapoint[0] : torch.tensor([array_img]) ; 
            reference : None.
            metric_config (example): {'bins' : 128}, 
        """
            
        if not torch.is_tensor(datapoint[0]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if metric_config is None or metric_config["bins"] is None:
            bins = 256
        else :
            bins = int(metric_config["bins"])
        
        arr = datapoint[0]
        
        norm_arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        hist, _ = np.histogram(norm_arr.flatten(), bins=bins, density=False)
        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-12)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))
    
    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0,
            description="Image entropy from a CT Scan image (NIFTI format)",
            value=np.array(data).mean(),
        )
        return res

class MeanGradientMagnitudeScale(StreamMetric):
    """
    Computes Mean Gradient Magnitude Scale of a CT Scan image (NIFTI format).
    We use 3 different scales to smooth the CT Scan, to help detect noise. 
    This metric takes into account noise level, image texture and image edges.
    Quantity of high-frequence content.
    
    Interpretation : 
        high value : a lot of details, maybe noise
        low value : smooth image
        
    By default : 
        sigma = 0.5 : little smoothing, get details 
        sigma = 1 : medium smoothing
        sigma = 2 : high smoothing, get large structures (ex : organs)
    """
    
    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requirement : 
        - source image in a tensor (datapoint[0])
        Optional :
        - tuples of sigmas values for smoothing (in metric_config["sigmas"])(default : (0.5, 1.0, 2.0))
        
        Raises :
            ValueError : if "datapoint[0]" is not a torch.tensor
        Args:
            datapoint : 
                datapoint[0] : torch.tensor([array_img]) ; 
            reference : None.
            metric_config (example): {'sigmas' : (0.5, 1.0, 2.0)}, 
        """
            
        if not torch.is_tensor(datapoint[0]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if metric_config is None or metric_config["sigmas"] is None:
            sigmas = (0.5, 1.0, 2.0)
        else :
            sigmas = metric_config["sigmas"]
        
        arr = datapoint[0]
        image_sitk = sitk.GetImageFromArray(arr)
        grads = []
        for s in sigmas:
            smoothed = sitk.DiscreteGaussian(image_sitk, s)
            grad = sitk.GradientMagnitude(smoothed)
            grads.append(float(np.mean(sitk.GetArrayFromImage((grad)))))
        return float(np.mean(grads))
        

    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0,
            description="Mean Gradient Magnitude with Scales from a CT Scan image (NIFTI format)",
            value=data,
        )
        return res
    

class _TaskTransferFunction_tools():
    """
    Toolkit for TaskTranferFunction classes
    """
    def compute_ttf_3d_fast(
        self,
        volume,
        spacing,
        max_profiles=1000,
        profile_half_length_mm=10.0,
        step_mm=0.25,
        sampling_step=3,
    ):
        Z, Y, X = volume.shape
        spacing = ast.literal_eval(spacing)
        sx, sy, sz = spacing
        # --- 1. gradient
        gz, gy, gx = np.gradient(volume, sz, sy, sx)
        grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        # --- 2. avoid flat zones
        threshold = np.percentile(grad_mag, 98)

        profiles = []
        
        sample_positions = np.arange(
            -profile_half_length_mm,
            profile_half_length_mm + step_mm,
            step_mm
            
        )
        
        margin = int(np.ceil(profile_half_length_mm) / min(spacing))

        # --- 3.  undersampled browsing
        for z in range(margin, Z - margin, sampling_step):
            for y in range(margin, Y - margin, sampling_step):
                for x in range(margin, X - margin, sampling_step):

                    if grad_mag[z, y, x] < threshold:
                        continue

                    direction = np.array([gz[z, y, x], gy[z, y, x], gx[z, y, x]])
                    norm = np.linalg.norm(direction)

                    if norm < 1e-8:
                        continue

                    direction = direction / norm

                    profile = []
                    valid = True
                    
                    for d_mm in sample_positions:
                        dz_mm = direction[0] * d_mm
                        dy_mm = direction[1] * d_mm
                        dx_mm = direction[2] * d_mm
                        # get indices from mm
                        zf = z + dz_mm / sz
                        yf = y + dy_mm / sy
                        xf = x + dx_mm / sx
                        
                        if (zf < 0 or zf >= Z-1 or yf < 0 or yf >= Y-1 or xf < 0 or xf >= X-1):
                            profile = None
                            break
                        
                        val = map_coordinates(volume, [[zf],[yf],[xf]], order=1, mode="nearest")[0]
                        profile.append(val)
                        
                    if profile is not None:
                        profiles.append(profile)

                    # stop early
                    if len(profiles) >= max_profiles:
                        break
                if len(profiles) >= max_profiles:
                    break
            if len(profiles) >= max_profiles:
                break

        if len(profiles) < 20:
            raise ValueError("Not enough profiles")

        profiles = np.array(profiles)

        # --- alignment
        aligned = []
        for p in profiles:
            g = np.gradient(p)
            center = np.argmax(np.abs(g))
            shift = len(p)//2 - center
            aligned.append(np.roll(p, shift))

        aligned = np.array(aligned)

        # --- ESF (Edge Spread Function) : how an ideal border is smoothed by the scanner
        esf = np.mean(aligned, axis=0)

        # light smoothing
        esf = np.convolve(esf, np.ones(3)/3, mode='same')

        # --- LSF (Edge Spread Function)
        lsf = np.gradient(esf,step_mm)

        # --- FFT : transform into frequencies
        ttf = np.abs(np.fft.fft(lsf))
        ttf /= np.max(ttf)

        freqs = np.fft.fftfreq(
            len(lsf), 
            d = step_mm
        )
        
        keep = freqs > 0
        freqs = freqs[keep]
        ttf = ttf[keep]

        return ttf, freqs

    def compute_ttf_metrics_interp(self, ttf, freqs, ttf_value):
        # interpolation

        ttf = np.array(ttf)
        freqs = np.array(freqs)

        def interp_x(target):
            # linear interpolation function

            # find where the curve cross the value
            for i in range(len(ttf) - 1):
                if (ttf[i] >= target and ttf[i+1] <= target):

                    # linear interpolation
                    x1, x2 = freqs[i], freqs[i+1]
                    y1, y2 = ttf[i], ttf[i+1]

                    return x1 + (target - y1) * (x2 - x1) / (y2 - y1)

            return np.nan  # if not found

        ttf_computed_val = interp_x(ttf_value)
        return ttf_computed_val

class ApproxTaskTransferFunction50(_TaskTransferFunction_tools, StreamMetric):
    """
    Computes anatomical approximation of Task Transfer Function 50 of a CT Scan image (NIFTI format), based on volume contouring.
    
    TTF50 and TTF10 are scalar metrics derived from the Task Transfer Function (TTF),
    which describes how well an imaging system preserves contrast at different
    spatial frequencies.

    - TTF50: Frequency at which the TTF falls to 50% of its maximum.
            Reflects effective spatial resolution (image sharpness).
            Low values : < 0.15
            High values : > 0.25

    Higher values indicate better resolution performance, while lower values
    suggest increased blurring or smoothing.

    These metrics should be interpreted alongside noise measurements (e.g., NoisePowerSpectrum)
    to assess overall image quality.
    """
    def compute_ttf_3d_fast(
        self,
        volume,
        spacing,
        max_profiles=1000,
        profile_half_length=10,
        step_mm=0.25,
        sampling_step=3,
    ):
        return super().compute_ttf_3d_fast(volume,
            spacing,
            max_profiles,
            profile_half_length,
            step_mm,
            sampling_step,
        )
    
    def compute_ttf_metrics_interp(self, ttf, freqs, ttf_value):
        return super().compute_ttf_metrics_interp(ttf, freqs, ttf_value)
    
    
    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requirement : 
        - base image in a tensor (datapoint[0])
        In a dictionary "metric_config" :
        - tuple img_spacing
        Optional :
        In a dictionary "metric_config" :
        - tuple img_spacing
        - int max_profiles
        - int profile_half_length
        - int sampling_step
        
        Raises :
            ValueError : if "datapoint[0]" is not a torch.tensor
            ValueError : if "img_spacing" is missing
        Args:
            datapoint : 
                datapoint[0] : torch.tensor([img]) ; 
            reference : None.
            metric_config (example): {'img_spacing':(0.66,0.66,1), 'max_profiles' : 1000, 'profile_half_length':10, 'sampling_step':3}
        """
        if not torch.is_tensor(datapoint[0]):
            raise ValueError("Data must be structured in a torch.tensor.")
        
        if metric_config is None or "img_spacing" not in metric_config:
            raise ValueError("Image spacing missing.")
        else :
            spacing = datapoint[2][str(metric_config["img_spacing"])]
            
        if metric_config is None or "step_mm" not in metric_config:
            step_mm = 0.25
        else :
            step_mm = float(metric_config["step_mm"])
        
        if metric_config is None or "max_profiles" not in metric_config:
            max_profiles = 1000
        else :
            max_profiles = int(metric_config["max_profiles"])
            
        if metric_config is None or "profile_half_length" not in metric_config:
            profile_half_length = 10
        else :
            profile_half_length = int(metric_config["profile_half_length"])
            
        if metric_config is None or "sampling_step" not in metric_config:
            sampling_step = 3
        else :
            sampling_step = int(metric_config["sampling_step"])
    
        arr = datapoint[0]
        
        ttf, freqs = self.compute_ttf_3d_fast(arr, 
                                        spacing=spacing, 
                                        max_profiles=max_profiles,
                                        profile_half_length=profile_half_length,
                                        step_mm=step_mm,
                                        sampling_step=sampling_step)
        

        ttf50 = self.compute_ttf_metrics_interp(ttf, freqs,0.5)
        return ttf50


    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0.15,
            description="Task Transfer Function 50 from a CT Scan image (NIFTI format)",
            value=np.array(data).mean(),
        )
        return res

class ApproxTaskTransferFunction10(_TaskTransferFunction_tools, StreamMetric):
    """
    Computes anatomical approximation of Task Transfer Function 10 of a CT Scan image (NIFTI format), based on volume contouring.
    
    TTF50 and TTF10 are scalar metrics derived from the Task Transfer Function (TTF),
    which describes how well an imaging system preserves contrast at different
    spatial frequencies.

    - TTF10: Frequency at which the TTF falls to 10% of its maximum.
            Reflects limiting resolution (visibility of fine details).
            Low values : < 0.08
            High values : > 0.15

    Higher values indicate better resolution performance, while lower values
    suggest increased blurring or smoothing.

    These metrics should be interpreted alongside noise measurements (e.g., NPS)
    to assess overall image quality.
    """
    def compute_ttf_3d_fast(
        self,
        volume,
        spacing,
        max_profiles=1000,
        profile_half_length=10,
        step_mm=0.25,
        sampling_step=3,
    ):
        return super().compute_ttf_3d_fast(volume,
            spacing,
            max_profiles,
            profile_half_length,
            step_mm,
            sampling_step,
        )
    
    def compute_ttf_metrics_interp(self, ttf, freqs, ttf_value):
        return super().compute_ttf_metrics_interp(ttf, freqs, ttf_value)
    
    
    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requirement : 
        - base image in a tensor (datapoint[0])
        In a dictionary "metric_config" :
        - tuple img_spacing
        
        Optional :
        In a dictionary "metric_config" :
        - int max_profiles
        - int profile_half_length
        - int sampling_step
        
        Raises :
            ValueError : if "datapoint[0]" is not a torch.tensor
        Args:
            datapoint : 
                datapoint[0] : torch.tensor([img]) ; 
            reference : None.
            metric_config (example): {'img_spacing':(0.66,0.66,1), 'max_profiles' : 1000, 'profile_half_length':10, 'sampling_step':3}
        """
        if not torch.is_tensor(datapoint[0]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if metric_config is None or metric_config["img_spacing"] is None:
            raise ValueError("Image spacing missing.")
        else:
            spacing = datapoint[2][str(metric_config["img_spacing"])]
        
        if metric_config is None or "step_mm" not in metric_config:
            step_mm = 0.25
        else :
            step_mm = float(metric_config["step_mm"])
            
        if metric_config is None or "max_profiles" not in metric_config:
            max_profiles = 1000
        else :
            max_profiles = int(metric_config["max_profiles"])
            
        if metric_config is None or "profile_half_length" not in metric_config:
            profile_half_length = 10
        else :
            profile_half_length = int(metric_config["profile_half_length"])
            
        if metric_config is None or "sampling_step" not in metric_config:
            sampling_step = 3
        else :
            sampling_step = int(metric_config["sampling_step"])
            
        arr = datapoint[0]
        
        ttf, freqs = self.compute_ttf_3d_fast(arr, 
                                        spacing=spacing, 
                                        max_profiles=max_profiles,
                                        profile_half_length=profile_half_length,
                                        step_mm=step_mm,
                                        sampling_step=sampling_step)
        

        ttf10 = self.compute_ttf_metrics_interp(ttf, freqs,0.1)
        return ttf10


    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster="Measurement Process",
            threshold=0.08,
            description="Task Transfer Function 10 from a CT SCan image (NIFTI format)",
            value=data,
        )
        return res
    
class _NoisePowerSpectrum_avg3D_tools():
    
    def _get_nps_2d(self, arr, patch_size, stride, keep_fraction, apply_window):
        H, W = arr.shape

        patches = []
        scores = []

        # --- Hanning Window
        if apply_window:
            win = np.hanning(patch_size)
            window = np.outer(win, win)
        else:
            window = np.ones((patch_size, patch_size))

        # --- 1. Extract patches
        for i in range(0, H - patch_size +1 , stride):
            for j in range(0, W - patch_size +1, stride):

                patch = arr[i:i+patch_size, j:j+patch_size]
                
                gx, gy = np.gradient(patch)
                grad = np.sqrt(gx**2 + gy**2)
                score = np.mean(grad)

                patches.append(patch)
                scores.append(score)

        patches = np.array(patches)
        scores = np.array(scores)

        if len(patches) == 0:
            raise ValueError("No patch extracted.")
        

        # --- 2. Select best patches
        N_keep = max(10, int(len(patches) * keep_fraction))
        idx = np.argsort(scores)[:N_keep]
        selected_patches = patches[idx]
    
        # --- 3. Compute NPS
        nps_list = []

        for patch in selected_patches:
            patch = patch - np.mean(patch)
            patch = patch * window

            fft = np.fft.fftshift(np.fft.fft2(patch))
            
            norm_factor = np.sum(window**2)
            nps = (np.abs(fft) ** 2) / norm_factor

            nps_list.append(nps)

        nps_2d_mean = np.mean(nps_list, axis=0)
        
        return nps_2d_mean
        
    def get_nps_1d_freq(self, arr, spacing_xy, patch_size, stride, keep_fraction, apply_window):

        if arr.ndim == 3:
            slices_tmp = arr
        else:
            slices_tmp = arr[None, ...]
            
        
        slice_indices = np.linspace(0, slices_tmp.shape[0]-1, 10, dtype=int)
        slices = slices_tmp[slice_indices]
        
        sx, sy = spacing_xy

        nps_list_all = []

        for slice_2d in slices:
            nps_2d_mean = self._get_nps_2d(slice_2d, patch_size, stride, keep_fraction, apply_window)
            nps_list_all.append(nps_2d_mean)

        nps_2d_mean = np.mean(nps_list_all, axis=0)

        # Radial averaging
        
        fx = np.fft.fftfreq(patch_size, d=sx)
        fy = np.fft.fftfreq(patch_size, d=sy)
        FX, FY = np.meshgrid(np.fft.fftshift(fx),np.fft.fftshift(fy))
        r = np.sqrt(FX**2 + FY**2)
        
        freq_bin = min(abs(fx[1] - fx[0]), abs(fy[1] - fy[0]))
        r_bin = np.round(r / freq_bin).astype(np.int32)
        
        tbin = np.bincount(r_bin.ravel(), weights = nps_2d_mean.ravel())
        nr = np.bincount(r_bin.ravel())

        nps_1d = tbin / np.maximum(nr, 1)
        freqs = np.arange(len(nps_1d)) / freq_bin
        
        return nps_1d, freqs

class TotalPower_NoisePowerSpectrum_avg3D(StreamMetric, _NoisePowerSpectrum_avg3D_tools):
    """
    Computes Total power of Noise Power Spectrum of a CT Scan image (NIFTI format).
    We use the mean value of 10 slices to approximate 3D.
    
    The NPS describes how noise is distributed across spatial frequencies,
    providing both the magnitude and texture of noise.

    Key outputs:
        - total_power: overall noise level (proportional to noise variance), describes the quantity of noise 

    Interpretation:
        - Higher total_power → high noise
        - Lower total_power → cleaner image

    The NPS is a standard metric in CT image quality assessment and should
    be interpreted together with resolution metrics (e.g., TTF).
    
    By default : 
        patch_size = 32
        stride = 16
        keep_fraction = 0.1
        apply_window = True
        
    """
    
    def get_nps_1d_freq(self, arr, xy_spacing, patch_size, stride, keep_fraction, apply_window):
        return super().get_nps_1d_freq(arr, xy_spacing, patch_size, stride, keep_fraction, apply_window)

    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requirement : 
        - base image in a tensor (datapoint[0])
        Optional :
        In a dictionary "metric_config" :
        - tuple img_spacing
        - int patch_size
        - int stride
        - float keep_fraction
        - bool apply_window
        
        Raises :
            ValueError : if "datapoint[0]" is not a torch.tensor
        Args:
            datapoint : 
                datapoint[0] : torch.tensor([array_img]) ; 
            reference : None.
            metric_config (example): {'img_spacing':(0.66,0.66,1), 'patch_size' : 32, 'stride':16, 'keep_fraction':0.1, 'apply_window':True }
            
        """
        if not torch.is_tensor(datapoint[0]):
            raise ValueError("Data must be structured in a torch.tensor.")
        
        if metric_config is None or metric_config["img_spacing"] is None:
            raise ValueError("Image spacing missing.")
        else:
            str_spacing = datapoint[2][str(metric_config["img_spacing"])]
            xy_spacing = ast.literal_eval(str_spacing)[:2]
            
        if metric_config is None or "patch_size" not in metric_config:
            patch_size = 32
        else :
            patch_size = int(metric_config["patch_size"])
            
        if metric_config is None or "stride" not in metric_config:
            stride = 16
        else :
            stride = int(metric_config["stride"])
            
        if metric_config is None or "keep_fraction" not in metric_config:
            keep_fraction = 0.1
        else :
            keep_fraction = float(metric_config["keep_fraction"])
            
        if metric_config is None or "apply_window" not in metric_config:
            apply_window = True
        else :
            apply_window = bool(metric_config["apply_window"])
            
        
        arr = datapoint[0]        
        nps_1d, freqs = self.get_nps_1d_freq(arr, xy_spacing, patch_size, stride, keep_fraction, apply_window)

        # --- Metrics
        total_power = float(np.sum(nps_1d))

        return total_power


    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster="Measurement Process",
            threshold=0,
            description="Total Power Noise Power Spectrum from a CT Scan image (NIFTI format)",
            value=data,
        )
        return res
    
class Entropy_NoisePowerSpectrum_avg3D(StreamMetric, _NoisePowerSpectrum_avg3D_tools):
    """
    Computes the entropy of Noise Power Spectrum of a CT Scan image (NIFTI format).
    We use the mean value of 10 slices to approximate 3D.
    
    The NPS describes how noise is distributed across spatial frequencies,
    providing both the magnitude and texture of noise.

    Key outputs:
        - NPS entropy: entropy of noise power spectrum, describes the distribution of noise 

    Interpretation:
        - High NPS entropy → noise spread in a lot of frequences
        - Low NPS entropy → noise concentrated in few frequences

    The NPS is a standard metric in CT image quality assessment and should
    be interpreted together with resolution metrics (e.g., TTF).
    
    By default : 
        patch_size = 32
        stride = 16
        keep_fraction = 0.1
        apply_window = True
        
    """
    
    def get_nps_1d_freq(self, arr, xy_spacing, patch_size, stride, keep_fraction, apply_window):
        return super().get_nps_1d_freq(arr, xy_spacing, patch_size, stride, keep_fraction, apply_window)

    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requirement : 
        - base image in a tensor (datapoint[0])
        Optional :
        In a dictionary "metric_config" :
        - tuple img_spacing
        - int patch_size
        - int stride
        - float keep_fraction
        - bool apply_window
        
        Raises :
            ValueError : if "datapoint[0]" is not a torch.tensor
        Args:
            datapoint : 
                datapoint[0] : torch.tensor([array_img]) ; 
            reference : None.
            metric_config (example): {'img_spacing':(0.66,0.66,1), 'patch_size' : 32, 'stride':16, 'keep_fraction':0.1, 'apply_window':True }
            
        """
        if not torch.is_tensor(datapoint[0]):
            raise ValueError("Data must be structured in a torch.tensor.")
        
        if metric_config is None or metric_config["img_spacing"] is None:
            raise ValueError("Image spacing missing.")
        else:
            str_spacing = datapoint[2][str(metric_config["img_spacing"])]
            xy_spacing = ast.literal_eval(str_spacing)[:2]
        
        if metric_config is None or "patch_size" not in metric_config:
            patch_size = 32
        else :
            patch_size = int(metric_config["patch_size"])
            
        if metric_config is None or "stride" not in metric_config:
            stride = 16
        else :
            stride = int(metric_config["stride"])
            
        if metric_config is None or "keep_fraction" not in metric_config:
            keep_fraction = 0.1
        else :
            keep_fraction = float(metric_config["keep_fraction"])
            
        if metric_config is None or "apply_window" not in metric_config:
            apply_window = True
        else :
            apply_window = bool(metric_config["apply_window"])
            
        
        arr = datapoint[0]

        nps_1d, freqs = self.get_nps_1d_freq(arr, xy_spacing, patch_size, stride, keep_fraction, apply_window)
        
        p = nps_1d / np.sum(nps_1d)
        p = p[p > 0]

        entropy_nps = -np.sum(p * np.log2(p))

        return entropy_nps


    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster="Measurement Process",
            threshold=0,
            description="Noise Power Spectrum Entropy from a CT Scan image (NIFTI format)",
            value=data,
        )
        return res
class DICESimilarityCoefficient(StreamMetric):
    """
    Computes de DSC between two segmentations.
    Needs to have two segmentation files in NIFTI format.
    In the dataset : segmentations are loaded with sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).
    """

    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requirement : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the required information about segmentations to the right columns in the metadata file. 
        Therefore, we need specific information about images in the metadata file. It is accessible with simpleITK : 
            img_seg1 = sitk.ReadImage(seg1_path)
            seg1_origin = img_seg1.GetOrigin()
            seg1_spacing = img_seg1.GetSpacing()
            seg1_direction = img_seg1.GetDirection()
        
        Raises :
            ValueError : if "datapoint[1]" is not a torch.tensor
            ValueError : if len(datapoint[1]) != 2 : datapoint[1] must contains seg1 and seg2
            ValueError : if ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] not in metric_config.keys() : metric_config must contains those keys
        Args:
            datapoint : 
                datapoint[1] : torch.tensor([seg1,seg2]) ; 
            reference : None.
            metric_config (example): {'seg1_origin' : 'col1', 'seg1_spacing':'col2', 'seg1_direction':'col3','seg2_origin' : 'col4', 'seg2_spacing':'col5', 'seg2_direction':'col6'}.
        """
        if not torch.is_tensor(datapoint[1]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if len(datapoint[1]) != 2 :
            raise ValueError("Tensor must contains : segmentation 1 (array), segmentation 2 (array)")
        if "seg1_origin" not in metric_config or "seg1_spacing" not in metric_config or "seg1_direction" not in metric_config  or \
            "seg2_origin" not in metric_config or "seg2_spacing" not in metric_config or "seg2_direction" not in metric_config :
            raise ValueError("metric_config must include the following keys : ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'].")
        
        overlap = sitk.LabelOverlapMeasuresImageFilter()
        
        seg1_img = sitk.GetImageFromArray(datapoint[1][0])
        
        # rebuild segmentation specifications
        seg1_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg1_origin"])])) # origin
        seg1_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg1_spacing"])])) # spacing
        seg1_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg1_direction"])])) # direction

        seg2_img = sitk.GetImageFromArray(datapoint[1][1])
        # rebuild segmentation specifications
        seg2_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg2_origin"])])) # origin
        seg2_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg2_spacing"])])) # spacing
        seg2_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg2_direction"])])) # direction

        overlap.Execute(seg1_img, seg2_img)
        dice = overlap.GetDiceCoefficient()
        return dice

    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster="Measurement Process",
            threshold=0.9,
            description="DICE",
            value=data,
        )
        return res

class IntersectionOverUnion(StreamMetric):
    """
    Computes de Intersection over Union score between two segmentations.
    Needs to have two segmentation files in NIFTI format.
    In the dataset : segmentations are loaded with sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).
    """

    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requirement : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the required information about segmentations to the right columns in the metadata file. 
        Therefore, we need specific information about images in the metadata file. It is accessible with simpleITK : 
            img_seg1 = sitk.ReadImage(seg1_path)
            seg1_origin = img_seg1.GetOrigin()
            seg1_spacing = img_seg1.GetSpacing()
            seg1_direction = img_seg1.GetDirection()
        
        Raises :
            ValueError : if "datapoint[1]" is not a torch.tensor
            ValueError : if len(datapoint[1]) != 2 : datapoint[1] must contains seg1 and seg2
            ValueError : if ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] not in metric_config.keys() : metric_config must contains those keys
        Args:
            datapoint : 
                datapoint[1] : torch.tensor([seg1,seg2]) ; 
            reference : None.
            metric_config (example): {'seg1_origin' : 'col1', 'seg1_spacing':'col2', 'seg1_direction':'col3','seg2_origin' : 'col4', 'seg2_spacing':'col5', 'seg2_direction':'col6'}.
        """
        if not torch.is_tensor(datapoint[1]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if len(datapoint[1]) != 2 :
            raise ValueError("Tensor must contains : segmentation 1 (array), segmentation 2 (array)")
        if "seg1_origin" not in metric_config or "seg1_spacing" not in metric_config or "seg1_direction" not in metric_config  or \
            "seg2_origin" not in metric_config or "seg2_spacing" not in metric_config or "seg2_direction" not in metric_config :
            raise ValueError("metric_config must include the following keys : ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'].")
        
        overlap = sitk.LabelOverlapMeasuresImageFilter()
        
        seg1_img = sitk.GetImageFromArray(datapoint[1][0])
        # rebuild segmentation specifications
        seg1_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg1_origin"])])) # origin
        seg1_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg1_spacing"])])) # spacing
        seg1_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg1_direction"])])) # direction

        seg2_img = sitk.GetImageFromArray(datapoint[1][1])
        # rebuild segmentation specifications
        seg2_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg2_origin"])])) # origin
        seg2_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg2_spacing"])])) # spacing
        seg2_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg2_direction"])])) # direction

        overlap.Execute(seg1_img, seg2_img)
        iou = overlap.GetJaccardCoefficient()
        return iou

    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster="Measurement Process",
            threshold=0.85,
            description="Intersection over Union Score between two segmentations",
            value=data,
        )
        return res

class _HausdorffDistance_tools():
    """
    Toolkit for HausdorffDistance classes
    """

    def _get_distances(self, seg1, seg2):

        # Distance maps (IMPORTANT: useImageSpacing=True → mm)
        seg1_dist = sitk.SignedMaurerDistanceMap(
            seg1, squaredDistance=False, useImageSpacing=True, insideIsPositive=False
        )
        seg2_dist = sitk.SignedMaurerDistanceMap(
            seg2, squaredDistance=False, useImageSpacing=True, insideIsPositive=False
        )

        # Surfaces (beaucoup plus robuste que ton extraction maison)
        seg1_surface = sitk.LabelContour(seg1)
        seg2_surface = sitk.LabelContour(seg2)

        # Conversion numpy
        seg1_surface_np = sitk.GetArrayViewFromImage(seg1_surface)
        seg2_surface_np = sitk.GetArrayViewFromImage(seg2_surface)

        seg1_dist_np = sitk.GetArrayViewFromImage(seg1_dist)
        seg2_dist_np = sitk.GetArrayViewFromImage(seg2_dist)

        # Distances surface → autre masque
        dists_seg1_to_seg2 = np.abs(seg2_dist_np[seg1_surface_np > 0])
        dists_seg2_to_seg1 = np.abs(seg1_dist_np[seg2_surface_np > 0])

        return dists_seg2_to_seg1, dists_seg1_to_seg2

class HausdorffDistance(_HausdorffDistance_tools,StreamMetric):
    """
    Computes Hausdorff Distance (in mm) between two segmentations.
    Needs to have two segmentation files in NIFTI format.
    In the dataset : segmentations are loaded with sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).
    """

    def _get_distances(self, seg1, seg2):
        """
        Gives distances between two segmentations, in mm

        Args:
            seg1 (sitk Image): first segmentation
            seg2 (sitk Image): second segmentation

        Returns:
            np.Array,np.Array : gives distances from seg1->seg2 and from seg2->seg1
        """
        

        return super()._get_distances(seg1, seg2)

    def _getHD(self, seg1, seg2):
        """ 
        Compute Hausdorff Distance (in mm) between two segmentations.

        Args:
            seg1 (sitk Image): first segmentation
            seg2 (sitk Image): second segmentation

        Returns:
            float: Hausdorff Distance
        """
        dists_seg2_to_seg1, dists_seg1_to_seg2 = self._get_distances(seg1, seg2)
        if len(dists_seg2_to_seg1) == 0 or len(dists_seg2_to_seg1) == 0 :
            raise ValueError("Error during computing : as least one segmentation is empty.")
        else:
            hd_max = max(dists_seg2_to_seg1.max(), dists_seg1_to_seg2.max())

        return hd_max

    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requirement : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the required information about segmentations to the right columns in the metadata file. 
        Therefore, we need specific information about images in the metadata file. It is accessible with simpleITK : 
            img_seg1 = sitk.ReadImage(seg1_path)
            seg1_origin = img_seg1.GetOrigin()
            seg1_spacing = img_seg1.GetSpacing()
            seg1_direction = img_seg1.GetDirection()
        
        Raises :
            ValueError : if "datapoint[1]" is not a torch.tensor
            ValueError : if len(datapoint[1]) != 2 : datapoint[1] must contains seg1 and seg2
            ValueError : if ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] not in metric_config.keys() : metric_config must contains those keys
        Args:
            datapoint : 
                datapoint[1] : torch.tensor([seg1,seg2]) ; 
            reference : None.
            metric_config (example): {'seg1_origin' : 'col1', 'seg1_spacing':'col2', 'seg1_direction':'col3','seg2_origin' : 'col4', 'seg2_spacing':'col5', 'seg2_direction':'col6'}.
        """
        if not torch.is_tensor(datapoint[1]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if len(datapoint[1]) != 2 :
            raise ValueError("Tensor must contains : segmentation 1 (array), segmentation 2 (array)")
        if "seg1_origin" not in metric_config or "seg1_spacing" not in metric_config or "seg1_direction" not in metric_config  or \
            "seg2_origin" not in metric_config or "seg2_spacing" not in metric_config or "seg2_direction" not in metric_config :
            raise ValueError("metric_config must include the following keys : ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'].")
        
        seg1_img = sitk.GetImageFromArray(datapoint[1][0])
        # rebuild segmentation specifications
        seg1_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg1_origin"])])) # origin
        seg1_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg1_spacing"])])) # spacing
        seg1_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg1_direction"])])) # direction

        seg2_img = sitk.GetImageFromArray(datapoint[1][1])
        # rebuild segmentation specifications
        seg2_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg2_origin"])])) # origin
        seg2_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg2_spacing"])])) # spacing
        seg2_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg2_direction"])])) # direction
        
        
        hd = self._getHD(seg1_img, seg2_img)
        return hd

    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster="Measurement Process",
            threshold=5.0,
            threshold_type="max",
            description="maximum Hausdorff Distance between two segmentations",
            value=data,
        )
        return res
    
class HausdorffDistance95(_HausdorffDistance_tools, StreamMetric):
    """
    Computes Hausdorff Distance 95 (in mm) between two segmentations.
    Needs to have two segmentation files in NIFTI format.
    In the dataset : segmentations are loaded with sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).
    """

    def _get_distances(self, seg1, seg2):
        """
        Gives distances between two segmentations, in mm

        Args:
            seg1 (sitk Image): first segmentation
            seg2 (sitk Image): second segmentation

        Returns:
            np.Array,np.Array : gives distances from seg1->seg2 and from seg2->seg1
        """
        

        return super()._get_distances(seg1, seg2)

    def _getHD95(self, seg1, seg2):
        """
        Compute Hausdorff Distance 95 (in mm) between two segmentations.
        HD95 removes 5% of extreme values compared to full Hausdorff Distance.

        Args:
            seg1 (sitk Image): first segmentation
            seg2 (sitk Image): second segmentation

        Returns:
            float: Hausdorff Distance 95
        """

        dists_seg2_to_seg1, dists_seg1_to_seg2 = self._get_distances(seg1, seg2)
        if len(dists_seg2_to_seg1) == 0 or len(dists_seg2_to_seg1) == 0 :
            # Hausdorff symmetric
            raise ValueError("Error during computing : as least one segmentation is empty.")
        else:
            hd95 = max(np.percentile(dists_seg2_to_seg1, 95), np.percentile(dists_seg1_to_seg2, 95))
        
        return hd95
            
    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requirement : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the required information about segmentations to the right columns in the metadata file. 
        Therefore, we need specific information about images in the metadata file. It is accessible with simpleITK : 
            img_seg1 = sitk.ReadImage(seg1_path)
            seg1_origin = img_seg1.GetOrigin()
            seg1_spacing = img_seg1.GetSpacing()
            seg1_direction = img_seg1.GetDirection()
        
        Raises :
            ValueError : if "datapoint[1]" is not a torch.tensor
            ValueError : if len(datapoint[1]) != 2 : datapoint[1] must contains seg1 and seg2
            ValueError : if ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] not in metric_config.keys() : metric_config must contains those keys
        Args:
            datapoint : 
                datapoint[1] : torch.tensor([seg1,seg2]) ; 
            reference : None.
            metric_config (example): {'seg1_origin' : 'col1', 'seg1_spacing':'col2', 'seg1_direction':'col3','seg2_origin' : 'col4', 'seg2_spacing':'col5', 'seg2_direction':'col6'}.
        """
        if not torch.is_tensor(datapoint[1]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if len(datapoint[1]) != 2 :
            raise ValueError("Tensor must contains : segmentation 1 (array), segmentation 2 (array)")
        if "seg1_origin" not in metric_config or "seg1_spacing" not in metric_config or "seg1_direction" not in metric_config  or \
            "seg2_origin" not in metric_config or "seg2_spacing" not in metric_config or "seg2_direction" not in metric_config :
            raise ValueError("metric_config must include the following keys : ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'].")
        
        seg1_img = sitk.GetImageFromArray(datapoint[1][0])
        # rebuild segmentation specifications
        seg1_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg1_origin"])])) # origin
        seg1_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg1_spacing"])])) # spacing
        seg1_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg1_direction"])])) # direction

        seg2_img = sitk.GetImageFromArray(datapoint[1][1])
        # rebuild segmentation specifications
        seg2_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg2_origin"])])) # origin
        seg2_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg2_spacing"])])) # spacing
        seg2_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg2_direction"])])) # direction
        
        
        hd95 = self._getHD95(seg1_img, seg2_img)
        return hd95

    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster="Measurement Process",
            threshold=1.0,
            threshold_type="max",
            description="Hausdorff Distance 95% between two segmentations",
            value=data,
        )
        return res


class DICEMean(DICESimilarityCoefficient, StreamMetric):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "dice_coefficient"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        DICESimilarityCoefficient.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.mean(data),
            cluster="Measurement Process",
            description="DICE mean",
        )

class DICEMedian(DICESimilarityCoefficient, StreamMetric):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "dice_coefficient"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        DICESimilarityCoefficient.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.median(data),
            cluster="Measurement Process",
            description="DICE median",
        )

class IntersectionOverUnionMean(IntersectionOverUnion, StreamMetric):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "intersection_over_union"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        IntersectionOverUnion.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.mean(data),
            cluster="Measurement Process",
            description="Intersection over Union mean",
        )

class IntersectionOverUnionMedian(IntersectionOverUnion, StreamMetric):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "intersection_over_union"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        IntersectionOverUnion.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.median(data),
            cluster="Measurement Process",
            description="Intersection over Union median",
        )
        
class HausdorffDistanceMean(HausdorffDistance, StreamMetric):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "hausdorff_distance"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        HausdorffDistance.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.mean(data),
            cluster="Measurement Process",
            description="HD mean",
        )

class HausdorffDistanceMedian(HausdorffDistance, StreamMetric):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "hausdorff_distance"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        HausdorffDistance.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.median(data),
            cluster="Measurement Process",
            description="HD median",
        )

class HausdorffDistance95Mean(HausdorffDistance95, StreamMetric):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "hausdorff_distance95"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        HausdorffDistance95.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.mean(data),
            cluster="Measurement Process",
            description="HD95 mean",
        )

class HausdorffDistance95Median(HausdorffDistance95, StreamMetric):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "hausdorff_distance95"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        HausdorffDistance95.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.median(data),
            cluster="Measurement Process",
            description="HD95 median",
        )
