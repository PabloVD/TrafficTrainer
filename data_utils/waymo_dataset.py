import os
import numpy as np
from torch.utils.data import Dataset


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

        # batch = {}
        # for key in data.keys():
        #     print(key, type(data[key]), data[key].shape)
        #     batch[key]=data[key]

        # batch["agent_ind"]=batch["agent_ind"].reshape(-1,1)
        # batch["scenario_id"]=batch["scenario_id"].reshape(-1,1)

        # print(batch["agent_ind"], batch["scenario_id"])

        
        # batch = {"x":data["raster"],
        #          "y": data["gt_marginal"],
        #          "is_available": data["future_val_marginal"],
        #          "XY":data["XY"]}

        # raster = data["raster"]#.astype("float32")

        # for key in data.keys():
        #     print(key)#,data["key".shape])

        # data = dict(data)
        # print(type(data))

        # print("er")
        # print(data["raster"].shape,data["XY"].shape)
        # print(data["agents_data"])
        # print(data["tl_data"])
        # print(data["roads_data"])



        # raster = data["raster"].astype("float32")
        
        # raster = raster.transpose(2, 1, 0) / 255

        # Temporarily commented
        # if self.is_test:
        #     center = data["shift"]
        #     yaw = data["yaw"]
        #     agent_id = data["object_id"]
        #     scenario_id = data["scenario_id"]

        #     return (
        #         raster,
        #         center,
        #         yaw,
        #         agent_id,
        #         str(scenario_id),
        #         data["_gt_marginal"],
        #         data["gt_marginal"],
        #     )

        # trajectory = data["gt_marginal"]

        # is_available = data["future_val_marginal"]

        # # if self.return_vector:
        # #     return raster, trajectory, is_available, data["vector_data"], data["center"], data["shift"], data["yaw"], str(data["scenario_id"]), data["gt_all"], data["val_all"]

        # return raster, trajectory, is_available

        return data
