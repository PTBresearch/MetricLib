import numpy as np
import torch
import antropy as ant
import SimpleITK as sitk
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
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
            x = np.asarray(datapoint[0][i, :], dtype=np.float64)
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


class DICESimilarityCoefficient(StreamMetric):
    """
    Computes de DSC between two segmentations.
    Needs to have two segmentation files in NIFTI format.
    In the dataset : segmentations are loaded with sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).
    """

    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the requiered information about segmentations to the right columns in the metadata file. 
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
        if metric_config["seg1_origin"] is None or metric_config["seg1_spacing"] is None or metric_config["seg1_direction"] is None or \
           metric_config["seg2_origin"] is None or metric_config["seg2_spacing"] is None or metric_config["seg2_direction"] is None :
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
            cluster=None,
            threshold=0,
            description="DICE Score between two segmentations",
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
        Requieres : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the requiered information about segmentations to the right columns in the metadata file. 
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
        if metric_config["seg1_origin"] is None or metric_config["seg1_spacing"] is None or metric_config["seg1_direction"] is None or \
           metric_config["seg2_origin"] is None or metric_config["seg2_spacing"] is None or metric_config["seg2_direction"] is None :
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
            cluster=None,
            threshold=0,
            description="Intersection over Union Score between two segmentations",
            value=data,
        )
        return res


class HausdorffDistance(StreamMetric):
    """
    Computes de Hausdorff Distance (in mm) between two segmentations.
    Needs to have two segmentation files in NIFTI format.
    In the dataset : segmentations are loaded with sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).
    """

    def _mask_to_surface_indices(self, mask_np):
        """
        Gives only surface mask from complete mask.
        Args:
            mask_np (np.Array): input mask, binary (z,y,x) array

        Returns:
            np.Array : indices in array index order (z,y,x)
        """
        eroded = binary_erosion(mask_np)
        surface = mask_np & (~eroded)
        inds = np.argwhere(surface)
        return inds

    def _indices_to_physical_points(self, img, inds):
        """
        Gives physical points (in mm) from indices in 3D sitk image
        Args:
            img (sitk Image): Segmentation image
            inds (np.Array): array Nx3 of (z,y,x), indices of surface voxels

        Returns:
            np.Array : shape (N,3) in mm (physical)
        """

        pts = [
            img.TransformIndexToPhysicalPoint((int(i[2]), int(i[1]), int(i[0])))
            for i in inds
        ]
        return np.array(pts)

    def _get_distances(self, seg1, seg2):
        """
        Gives distances between two segmentations, in mm

        Args:
            seg1 (sitk Image): first segmentation
            seg2 (sitk Image): second segmentation

        Returns:
            np.Array,np.Array : gives distances from seg1->seg2 and from seg2->seg1
        """
        seg1_np = sitk.GetArrayFromImage(seg1)  # (z,y,x)
        seg2_np = sitk.GetArrayFromImage(seg2)

        # surfaces indexes
        seg1_surf_idx = self._mask_to_surface_indices(seg1_np)
        seg2_surf_idx = self._mask_to_surface_indices(seg2_np)

        # physical points
        seg1_surf_pts = self._indices_to_physical_points(seg1, seg1_surf_idx)
        seg2_surf_pts = self._indices_to_physical_points(seg2, seg2_surf_idx)

        # KDTree distances seg2->seg1
        tree_seg1 = cKDTree(seg1_surf_pts)
        dists_seg2_to_seg1, idxs = tree_seg1.query(seg2_surf_pts, k=1)
        # KDTree distances seg1->seg2
        tree_seg2 = cKDTree(seg2_surf_pts)
        dists_seg1_to_seg2, idxs2 = tree_seg2.query(seg1_surf_pts, k=1)

        return dists_seg2_to_seg1, dists_seg1_to_seg2

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
        hd_max = max(dists_seg2_to_seg1.max(), dists_seg1_to_seg2.max())

        return hd_max

    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the requiered information about segmentations to the right columns in the metadata file. 
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
        if metric_config["seg1_origin"] is None or metric_config["seg1_spacing"] is None or metric_config["seg1_direction"] is None or \
           metric_config["seg2_origin"] is None or metric_config["seg2_spacing"] is None or metric_config["seg2_direction"] is None :
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
        
        
        hd95 = self._getHD(seg1_img, seg2_img)
        return hd95

    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0,
            description="maximum Hausdorff Distance between two segmentations",
            value=data,
        )
        return res
    
class HausdorffDistance95(StreamMetric):
    """
    Computes de Hausdorff Distance 95 (in mm) between two segmentations.
    Needs to have two segmentation files in NIFTI format.
    In the dataset : segmentations are loaded with sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).
    """

    def _mask_to_surface_indices(self, mask_np):
        """
        Gives only surface mask from complete mask.
        Args:
            mask_np (np.Array): input mask, binary (z,y,x) array

        Returns:
            np.Array : indices in array index order (z,y,x)
        """
        eroded = binary_erosion(mask_np)
        surface = mask_np & (~eroded)
        inds = np.argwhere(surface)
        return inds

    def _indices_to_physical_points(self, img, inds):
        """
        Gives physical points (in mm) from indices in 3D sitk image
        Args:
            img (sitk Image): Segmentation image
            inds (np.Array): array Nx3 of (z,y,x), indices of surface voxels

        Returns:
            np.Array : shape (N,3) in mm (physical)
        """

        pts = [
            img.TransformIndexToPhysicalPoint((int(i[2]), int(i[1]), int(i[0])))
            for i in inds
        ]
        return np.array(pts)

    def _get_distances(self, seg1, seg2):
        """
        Gives distances between two segmentations, in mm

        Args:
            seg1 (sitk Image): first segmentation
            seg2 (sitk Image): second segmentation

        Returns:
            np.Array,np.Array : gives distances from seg1->seg2 and from seg2->seg1
        """
        seg1_np = sitk.GetArrayFromImage(seg1)  # (z,y,x)
        seg2_np = sitk.GetArrayFromImage(seg2)

        # surfaces indexes
        seg1_surf_idx = self._mask_to_surface_indices(seg1_np)
        seg2_surf_idx = self._mask_to_surface_indices(seg2_np)

        # physical points
        seg1_surf_pts = self._indices_to_physical_points(seg1, seg1_surf_idx)
        seg2_surf_pts = self._indices_to_physical_points(seg2, seg2_surf_idx)

        # KDTree distances seg2->seg1
        tree_seg1 = cKDTree(seg1_surf_pts)
        dists_seg2_to_seg1, idxs = tree_seg1.query(seg2_surf_pts, k=1)
        # KDTree distances seg1->seg2
        tree_seg2 = cKDTree(seg2_surf_pts)
        dists_seg1_to_seg2, idxs2 = tree_seg2.query(seg1_surf_pts, k=1)

        return dists_seg2_to_seg1, dists_seg1_to_seg2

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

        # Hausdorff symmetric
        hd95 = max(np.percentile(dists_seg2_to_seg1, 95), np.percentile(dists_seg1_to_seg2, 95))
        
        return hd95
            
    
    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the requiered information about segmentations to the right columns in the metadata file. 
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
        if metric_config["seg1_origin"] is None or metric_config["seg1_spacing"] is None or metric_config["seg1_direction"] is None or \
           metric_config["seg2_origin"] is None or metric_config["seg2_spacing"] is None or metric_config["seg2_direction"] is None :
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
            cluster=None,
            threshold=0,
            description="Hausdorff Distance 95% \between two segmentations",
            value=data,
        )
        return res
