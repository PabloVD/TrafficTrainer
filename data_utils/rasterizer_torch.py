# TO DO: 
# - Create class with all methods and attributes, e.g. self.batchsize
import cv2
import numpy as np
import torch

MAX_PIXEL_VALUE = 255


# Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
# Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
state_to_color = {1:"red",4:"red",7:"red",2:"yellow",5:"yellow",8:"yellow",3:"green",6:"green"}
color_to_rgb = { "red":(255,0,0), "yellow":(255,255,0),"green":(0,255,0) }

raster_size = 224

#--- Utils

def get_rotation_matrix(angle):

    c = torch.cos(angle)
    s = torch.sin(angle)
    rot_matrix = torch.stack([torch.stack([c, -s],dim=1),torch.stack([s, c],dim=1)],dim=1)   # shape: (batch_size, 2, 2)
    return rot_matrix

class Rasterizer():
     
    def __init__(self, zoom_fact):
         
        self.zoom_fact = zoom_fact

    #--- Roads and traffic lights routines

    def draw_roads(self, roadmap, centered_roadlines, roads_ids, tl_dict):

        unique_road_ids = np.unique(roads_ids)
        for road_id in unique_road_ids:
            if road_id >= 0:
                roadline = centered_roadlines[roads_ids == road_id]
                #road_type = roads_types[roads_ids == road_id].flatten()[0]

                #road_color = road_colors[road_type]
                road_color = 0
                for col in color_to_rgb.keys():
                    if road_id in tl_dict[col]:
                        road_color = color_to_rgb[col]
                #road_color = 0

                rd = np.array( roadline, dtype=int)
                # rd = roadline.to(torch.int)

                roadmap = cv2.polylines(
                    roadmap,
                    [rd],
                    False,
                    road_color
                )

        return roadmap

    def get_tl_dict(self, tl_states, tl_ids, tl_valids):

        tl_dict = {"green": set(), "yellow": set(), "red": set()}

        # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
        # Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
        for tl_state, tl_id, tl_valid in zip(tl_states.flatten(), tl_ids, tl_valids.flatten()):
            if tl_valid == 0 or tl_state==0:
                continue
            # print(tl_id, tl_state, tl_dict)
            # print(state_to_color[tl_state.item()])
            tl_dict[state_to_color[tl_state.item()]].add(tl_id)

        return tl_dict

    def get_road_map(self, centered_roadlines, roads_ids, tl_states_curr, tl_ids, tl_valid_curr):

        roadmaps = []
        for ag in range(self.batchsize):

            tl_dict = self.get_tl_dict(tl_states_curr[ag], tl_ids[ag], tl_valid_curr[ag])

            RES_ROADMAP = (np.ones((raster_size, raster_size, 3), dtype=np.uint8) * MAX_PIXEL_VALUE)
            RES_ROADMAP = self.draw_roads(RES_ROADMAP, centered_roadlines[ag], roads_ids[ag], tl_dict)
            roadmaps.append(RES_ROADMAP)

        roadmaps = np.array(roadmaps).transpose(0, 3, 2, 1)
        roadmaps = torch.tensor(roadmaps,device=self.device)

        return roadmaps
    
    #--- Agents

    def get_bounding_boxes(self, centered_others, angle_agents, lengths, widths):
         
        box_points_in = torch.cat([
                            -lengths,
                            -widths,
                            lengths,
                            -widths,
                            lengths,
                            widths,
                            -lengths,
                            widths
                            ],dim=-1)*self.zoom_fact/ 2
        box_points_in = box_points_in.reshape(self.batchsize, self.maxags, 4, 2)

        boxptstot = torch.zeros((self.batchsize, self.maxags, self.n_channels, 4, 2),device=self.device)
        
        for it in range(self.n_channels):
            rot_matrix_past = get_rotation_matrix(angle_agents[:,:,it].view(-1))
            box_points = torch.bmm(box_points_in.view(-1,4,2) , rot_matrix_past)        
            box_points = box_points + centered_others[:,:,it].view(-1,1,2)
            box_points = box_points.reshape(self.batchsize, self.maxags, 4, 2)
            boxptstot[:,:,it] = box_points

        return boxptstot

    def get_agents_map(self, centered_others, angle_agents, lengths, widths, agind):

        RES_EGO = np.zeros((self.batchsize, self.n_channels, raster_size, raster_size, 1) , dtype=np.uint8)
        RES_OTHER = np.zeros((self.batchsize, self.n_channels, raster_size, raster_size, 1) , dtype=np.uint8)

        boxptstot = self.get_bounding_boxes(centered_others, angle_agents, lengths, widths)

        boxptstot = boxptstot.cpu().detach().numpy()
        agind_np = agind.cpu().detach().numpy()
        boxptstot = boxptstot.astype(np.int32)

        for it in range(self.n_channels):
            box_points = boxptstot[:,:,it]
            box_ego = box_points[np.arange(self.batchsize), agind_np]

            for ag in range(self.batchsize):

                cv2.fillPoly(
                    RES_EGO[ag,it],
                    box_ego[ag].reshape(1,-1,2),
                    color=MAX_PIXEL_VALUE
                    )
                cv2.fillPoly(
                    RES_OTHER[ag,it],
                    box_points[ag],
                    color=MAX_PIXEL_VALUE
                    )
            
        RES_EGO = RES_EGO.transpose(0, 1, 3, 2, 4)
        RES_OTHER = RES_OTHER.transpose(0, 1, 3, 2, 4)
        RES_EGO = torch.tensor(RES_EGO,device=self.device).squeeze(-1)
        RES_OTHER = torch.tensor(RES_OTHER,device=self.device).squeeze(-1)                
        RES_OTHER = RES_OTHER - RES_EGO

        return RES_EGO, RES_OTHER


    # def get_ground_truth():
     
    def get_rasterized_input(self, pos_ego_curr, XY, rot_matrix, angle_agents, lengths, widths, agind, roads_coords, roads_ids, tl_states_curr, tl_ids, tl_valid_curr):
             
        displacement = torch.tensor([[raster_size // 4, raster_size // 2]], device=self.device)

        # Road coordinates
        centered_roadlines = torch.bmm( (roads_coords - pos_ego_curr.view(-1,1,2)) , rot_matrix )*self.zoom_fact + displacement
        centered_roadlines = centered_roadlines.cpu().detach().numpy()
        roads_ids = roads_ids.cpu().detach().numpy()

        # Road map
        roadmaps = self.get_road_map(centered_roadlines, roads_ids, tl_states_curr, tl_ids, tl_valid_curr)

        # Agent coordinates
        self.maxags = XY.shape[1]
        centered_others = torch.bmm( (XY.reshape(self.batchsize,-1,2) - pos_ego_curr.view(-1,1,2)) , rot_matrix)*self.zoom_fact + displacement
        centered_others = centered_others.view(self.batchsize, self.maxags, -1, 2)

        # Agents map
        RES_EGO, RES_OTHER = self.get_agents_map(centered_others, angle_agents, lengths, widths, agind)

        # Combine maps
        raster = torch.cat([roadmaps, RES_EGO, RES_OTHER],dim=1)
        raster = raster / 255 
        raster = raster.to(torch.float32)
    
        return raster

    #--- Main method to get raster input and ground truth for training

    def get_training_dict(self,
            ind_curr,
            batch,
            XY_in,
            YAWS_in,
            n_channels,
            device="cpu",
            ):
                
        agents_valid_in = batch["agents_valid"]
        lengths = batch["lengths"]
        widths = batch["widths"]

        roads_ids = batch["roads_ids"]
        roads_coords = batch["roads_coords"]

        tl_ids = batch["tl_ids"]
        tl_valid_in = batch["tl_valid"]
        tl_states_in = batch["tl_states"]

        agind = batch["agent_ind"]
        
        self.batchsize = XY_in.shape[0]
        self.n_channels = n_channels
        self.device = device
        btchrng = torch.arange(self.batchsize)

        XY = XY_in[:,:,ind_curr-self.n_channels+1:ind_curr+1]
        YAWS = YAWS_in[:,:,ind_curr-self.n_channels+1:ind_curr+1]
        GT_XY = XY_in[:,:,ind_curr+1:]
        future_yaw_all = YAWS_in[:,:,ind_curr+1:]
        future_valid = agents_valid_in[:,:,ind_curr+1:]

        tl_valid_hist = tl_valid_in[:,:,ind_curr-self.n_channels+1:ind_curr+1]
        tl_states_hist = tl_states_in[:,:,ind_curr-self.n_channels+1:ind_curr+1]
        tl_states_curr, tl_valid_curr = tl_states_hist[:,:,-1], tl_valid_hist[:,:,-1]
        tl_states_curr, tl_ids, tl_valid_curr = tl_states_curr.cpu().detach().numpy(), tl_ids.cpu().detach().numpy(), tl_valid_curr.cpu().detach().numpy()

        xy_val = XY[btchrng, agind]
        yaw_ego = YAWS[btchrng, agind]
        gt_xy = GT_XY[btchrng, agind]
        future_val = future_valid[btchrng, agind]
        future_yaw_ego = future_yaw_all[btchrng, agind]

        pos_ego_curr = xy_val[:,-1]
        yaw_ego_curr = yaw_ego[:,-1]

        rot_matrix = get_rotation_matrix(yaw_ego_curr)
        angle_agents = yaw_ego_curr.view(-1,1,1)-YAWS

        raster = self.get_rasterized_input(pos_ego_curr, XY, rot_matrix, angle_agents, lengths, widths, agind, roads_coords, roads_ids, tl_states_curr, tl_ids, tl_valid_curr)

        # Ground truth data
        centered_gt = torch.bmm( gt_xy - pos_ego_curr.view(-1,1,2) , rot_matrix )
        future_yaw = future_yaw_ego-yaw_ego_curr.view(-1,1)
        centered_gt = torch.cat([centered_gt,future_yaw.unsqueeze(-1)],dim=-1)

        raster_dict = {
            "raster": raster,
            "gt_marginal": centered_gt,
            "future_val_marginal": future_val,
        }

        return raster_dict