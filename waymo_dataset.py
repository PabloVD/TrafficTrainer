import os
import numpy as np
from torch.utils.data import Dataset

#fixed_frame = True
fixed_frame = False
#use_rgb = False
use_rgb = True

class WaymoLoader(Dataset):
    def __init__(self, directory, limit=0, return_vector=False, is_test=False):
        files = os.listdir(directory)
        self.files = [os.path.join(directory, f) for f in files if f.endswith(".npz")]

        if limit > 0:
            self.files = self.files[:limit]
        else:
            self.files = sorted(self.files)

        self.return_vector = return_vector
        self.is_test = is_test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        
        data = np.load(filename, allow_pickle=True)

        raster = data["raster"].astype("float32")
        
        if use_rgb and not fixed_frame:
            raster = raster.transpose(0, 3, 1, 2) / 255
            raster = raster.reshape(-1, raster.shape[-2], raster.shape[-1])
        else:
            raster = raster.transpose(2, 1, 0) / 255

        if self.is_test:
            center = data["shift"]
            yaw = data["yaw"]
            agent_id = data["object_id"]
            scenario_id = data["scenario_id"]

            return (
                raster,
                center,
                yaw,
                agent_id,
                str(scenario_id),
                data["_gt_marginal"],
                data["gt_marginal"],
            )

        trajectory = data["gt_marginal"]

        is_available = data["future_val_marginal"]

        if self.return_vector:
            return raster, trajectory, is_available, data["vector_data"], data["center"], data["shift"], data["yaw"], str(data["scenario_id"]), data["gt_all"], data["val_all"]

        return raster, trajectory, is_available
