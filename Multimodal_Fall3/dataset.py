import torch
import numpy as np
import numpy.typing as npt
from typing import List,Tuple

from torch.utils.data import Dataset

class GeneralDataset(Dataset):
    def __init__(self,samples:List[Tuple[npt.ArrayLike, npt.ArrayLike]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        skeleton, label = self.samples[idx]
        skeleton, label = torch.from_numpy(skeleton).to(torch.float32),torch.from_numpy(label).to(torch.float32)
        skeleton = skeleton.permute(2,0,1)

        return skeleton, label

class Fall2Dataset(GeneralDataset):
    def __init__(self,samples:List[Tuple[npt.ArrayLike, npt.ArrayLike]]):
        self.sample_strategy = None
        self.samples = samples

    def _scale_pose(self,xy):
        """
        Normalize pose points by scale with max/min value of each pose.
        xy : (frames, parts, xy) or (parts, xy)
        """
        if xy.ndim == 2:
            xy = np.expand_dims(xy, 0)
        xy_min = np.nanmin(xy, axis=1)
        xy_max = np.nanmax(xy, axis=1)
        for i in range(xy.shape[0]):
            xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
            xy[i] = np.nan_to_num(xy[i], copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        return xy.squeeze()

    def __getitem__(self,idx):
        skeleton, label = self.samples[idx]
        # Scale pose normalize.
        skeleton[:, :, :2] = self._scale_pose(skeleton[:, :, :2])
        # Add center point.
        skeleton = np.concatenate((skeleton, np.expand_dims((skeleton[:, 1, :] + skeleton[:, 2, :]) / 2, 1)), axis=1)
        skeleton, label = torch.from_numpy(skeleton).to(torch.float32),torch.from_numpy(label).to(torch.float32)
        skeleton = skeleton.permute(2,0,1)

        return skeleton, label