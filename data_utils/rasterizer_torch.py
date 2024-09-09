import cv2
import numpy as np
import torch

MAX_PIXEL_VALUE = 255
N_ROADS = 21
# road_colors = torch.arange(1, MAX_PIXEL_VALUE, N_ROADS, dtype=torch.int)

# Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
# Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
state_to_color = {1:"red",4:"red",7:"red",2:"yellow",5:"yellow",8:"yellow",3:"green",6:"green"}
color_to_rgb = { "red":(255,0,0), "yellow":(255,255,0),"green":(0,255,0) }

raster_size = 224
zoom_fact = 3.

def draw_roads(roadmap, centered_roadlines, roads_ids, tl_dict):

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

def get_tl_dict(tl_states, tl_ids, tl_valids):

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

def get_rotation_matrix(angle):

    c = torch.cos(angle)
    s = torch.sin(angle)
    rot_matrix = torch.stack([torch.stack([c, -s],dim=1),torch.stack([s, c],dim=1)],dim=1)   # shape: (batch_size, 2, 2)
    return rot_matrix


def rasterizer_torch(
        ind_curr,
        batch,
        XYin,
        YAWSin,
        n_channels,
        device="cpu",
        zoom_fact=zoom_fact,
        ):
    
    ag = batch["agent_ind"]
        
    agents_ids = batch["agents_ids"]
    agents_valid_split = batch["agents_valid"]
    XY_split = XYin
    YAWS_split = YAWSin
    lengths = batch["lengths"]
    widths = batch["widths"]
    
    roads_ids = batch["roads_ids"]
    roads_valid = batch["roads_valid"]
    roads_coords = batch["roads_coords"]

    tl_ids = batch["tl_ids"]
    tl_valid_split = batch["tl_valid"]
    tl_states_split = batch["tl_states"]

    batchsize = XY_split.shape[0]
    btchrng = torch.arange(batchsize)

    XY = XY_split[:,:,ind_curr-n_channels+1:ind_curr+1]
    GT_XY = XY_split[:,:,ind_curr+1:]
    YAWS = YAWS_split[:,:,ind_curr-n_channels+1:ind_curr+1]
    agents_valid = agents_valid_split[:,:,ind_curr-n_channels+1:ind_curr+1]
    future_valid = agents_valid_split[:,:,ind_curr+1:]

    tl_valid_hist = tl_valid_split[:,:,ind_curr-n_channels+1:ind_curr+1]
    tl_states_hist = tl_states_split[:,:,ind_curr-n_channels+1:ind_curr+1]

    displacement = torch.tensor([[raster_size // 4, raster_size // 2]], device=device)

    future_val = future_valid
        
    
    RES_EGO = np.zeros((batchsize, n_channels, raster_size, raster_size, 1) , dtype=np.uint8)
    RES_OTHER = np.zeros((batchsize, n_channels, raster_size, raster_size, 1) , dtype=np.uint8)
    # RES_EGO = [
    #     np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
    #     for _ in range(n_channels)
    # ]
    # RES_OTHER = [
    #     np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
    #     for _ in range(n_channels)
    # ]

    agind = batch["agent_ind"]

    xy = XY[btchrng, agind]
    yaw_ego = YAWS[btchrng, agind]
    ego_id = agents_ids[btchrng, agind]
    gt_xy = GT_XY[btchrng, agind]
    
    xy_val = xy

    # print(XY.shape, xy_val.shape)
    # roads_coords = roads_coords[roads_valid > 0]
    # roads_ids = roads_ids[roads_valid > 0]

    unscaled_center_xy = xy_val[:,-1]
    yawt = yaw_ego[:,-1]

    rot_matrix = get_rotation_matrix(yawt)

    centered_roadlines = torch.bmm( (roads_coords - unscaled_center_xy.view(-1,1,2)) , rot_matrix )*zoom_fact + displacement
    
    maxags = XY.shape[1]
    centered_others = torch.bmm( (XY.reshape(batchsize,-1,2) - unscaled_center_xy.view(-1,1,2)) , rot_matrix)*zoom_fact + displacement

    centered_others = centered_others.view(batchsize, maxags, -1, 2)
    # centered_others = centered_others.reshape(XY.shape[0], n_channels, 2)
    centered_gt = torch.bmm( gt_xy - unscaled_center_xy.view(-1,1,2) , rot_matrix )

    

    #print(centered_roadlines.shape, roads_ids.shape, tl_states_hist.shape, tl_ids.shape, tl_valid_hist.shape)

    centered_roadlines = centered_roadlines.cpu().detach().numpy()
    roads_ids = roads_ids.cpu().detach().numpy()

    roadmaps = []
    for ag in range(batchsize):

        tl_dict = get_tl_dict(tl_states_hist[ag,:,-1], tl_ids[ag], tl_valid_hist[ag,:,-1])

        RES_ROADMAP = (np.ones((raster_size, raster_size, 3), dtype=np.uint8) * MAX_PIXEL_VALUE)
        RES_ROADMAP = draw_roads(RES_ROADMAP, centered_roadlines[ag], roads_ids[ag], tl_dict)
        roadmaps.append(RES_ROADMAP)

    # Agents
    # unique_agent_ids = torch.unique(agents_ids)

    agent_l = lengths
    agent_w = widths

    box_points = torch.cat([
                        -agent_l,
                        -agent_w,
                        agent_l,
                        -agent_w,
                        agent_l,
                        agent_w,
                        -agent_l,
                        agent_w
                        ],dim=-1)*zoom_fact/ 2
    box_points = box_points.reshape(batchsize, maxags, 4, 2)


    ang = yaw_ego[:,-1].view(-1,1,1)-YAWS

    boxptstot = torch.zeros((batchsize, maxags, n_channels, 4, 2),device=device)
    
    for it in range(n_channels):
        rot_matrix_past = get_rotation_matrix(ang[:,:,it].view(-1))
        #print(rot_matrix_past.shape)
        box_points = torch.bmm(box_points.view(-1,4,2) , rot_matrix_past)
        
        box_points = box_points + centered_others[:,:,it].view(-1,1,2)
        box_points = box_points.reshape(batchsize, maxags, 4, 2)
        boxptstot[:,:,it] = box_points


    #box_ego = boxptstot[btchrng, agind]

    
    boxptstot = boxptstot.cpu().detach().numpy()
    agind_np = agind.cpu().detach().numpy()
    boxptstot = boxptstot.astype(np.int32)

    
    for it in range(n_channels):
        box_points = boxptstot[:,:,it]
        box_ego = box_points[np.arange(batchsize), agind_np]
        # box_others = boxptstot[np.arange(batchsize), ~agind_np]
        #print(box_ego.shape, box_points.shape)

        for ag in range(batchsize):
            # print(box_ego[ag].shape, box_points[ag].shape)            

            cv2.fillPoly(
                RES_EGO[ag,it],
                box_ego[ag].reshape(1,-1,2),
                color=MAX_PIXEL_VALUE
                )
            #for agg in range(128):
            cv2.fillPoly(
                RES_OTHER[ag,it],
                box_points[ag],
                color=MAX_PIXEL_VALUE
                )
            
    RES_OTHER = RES_OTHER - RES_EGO


    # is_ego = False
    # # self_type = 0
    # for other_agent_id in unique_agent_ids:
    #     other_agent_id = int(other_agent_id)
    #     if other_agent_id < 1:
    #         continue
    #     if other_agent_id == ego_id:
    #         is_ego = True
    #         # self_type = agent_type[agents_ids == other_agent_id]
    #     else:
    #         is_ego = False

    #     agent_lane = centered_others[agents_ids == other_agent_id][0]
    #     agent_valid = agents_valid[agents_ids == other_agent_id]
    #     agent_yaw = YAWS[agents_ids == other_agent_id]

    #     agent_l = lengths[agents_ids == other_agent_id]
    #     agent_w = widths[agents_ids == other_agent_id]

    #     for timestamp, (coord, valid_coordinate, past_yaw,) in enumerate(
    #         zip(
    #             agent_lane,
    #             agent_valid.flatten(),
    #             agent_yaw.flatten(),
    #         )
    #     ):
    #         if valid_coordinate == 0:
    #             continue


    #         _coord = coord.unsqueeze(0)
            
    #         yawt = yaw_ego[:,-1]            

    #         rot_matrix_past = get_rotation_matrix(yawt - past_yaw)
    #         box_points = box_points @ rot_matrix_past

    #         box_points = box_points + _coord
    #         box_points = box_points.reshape(1, -1, 2).to(torch.int32)

    #         box_points = box_points.cpu().detach().numpy()


    #         if is_ego:
    #             cv2.fillPoly(
    #                 RES_EGO[timestamp],
    #                 box_points,
    #                 color=MAX_PIXEL_VALUE
    #             )
    #         else:
    #             cv2.fillPoly(
    #                 RES_OTHER[timestamp],
    #                 box_points,
    #                 color=MAX_PIXEL_VALUE
    #             )

    roadmaps = np.array(roadmaps).transpose(0, 3, 2, 1)
    roadmaps = torch.tensor(roadmaps,device=device)
    RES_EGO = torch.tensor(RES_EGO,device=device).squeeze(-1)
    RES_OTHER = torch.tensor(RES_OTHER,device=device).squeeze(-1)
    raster = torch.cat([roadmaps, RES_EGO, RES_OTHER],dim=1)

    # raster = np.concatenate([RES_ROADMAP] + RES_EGO + RES_OTHER, axis=2)   
    # raster = raster.transpose(2, 1, 0) / 255 
    # raster = torch.tensor(raster, device=device)
    raster = raster / 255 
    raster = raster.to(torch.float32)

    raster_dict = {
        "object_id": ego_id,
        "raster": raster,
        "yaw": yaw_ego,
        "shift": unscaled_center_xy,
        "_gt_marginal": gt_xy,
        "gt_marginal": centered_gt,
        "future_val_marginal": future_val,
        "agent_ind": ag,
        "xy_ag": XY[ag]
        # "gt_joint": GT_XY[tracks_to_predict.flatten() > 0],
        # "future_val_joint": future_valid[tracks_to_predict.flatten() > 0],
        # "scenario_id": scenario_id,
        # "self_type": self_type,
    }

    return raster_dict