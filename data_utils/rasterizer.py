import cv2
import numpy as np

MAX_PIXEL_VALUE = 255
N_ROADS = 21
road_colors = [int(x) for x in np.linspace(1, MAX_PIXEL_VALUE, N_ROADS).astype("uint8")]

# Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
# Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
state_to_color = {1:"red",4:"red",7:"red",2:"yellow",5:"yellow",8:"yellow",3:"green",6:"green"}
color_to_rgb = { "red":(255,0,0), "yellow":(255,255,0),"green":(0,255,0) }

raster_size = 224

displacement = np.array([[raster_size // 4, raster_size // 2]])

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

            roadmap = cv2.polylines(
                roadmap,
                [roadline.astype(int)],
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
        tl_dict[state_to_color[tl_state]].add(tl_id)

    return tl_dict

def rasterizer(
        ag,
        ind_curr,
        agents_data,
        roads_data,
        tl_data,
        n_channels,
        zoom_fact):
    
    agents_ids = agents_data["agents_ids"]
    agents_valid_split = agents_data["agents_valid"]
    XY_split = agents_data["XY"]
    YAWS_split = agents_data["YAWS"]
    lengths = agents_data["lengths"]
    widths = agents_data["widths"]
    
    roads_ids = roads_data["roads_ids"]
    roads_coords = roads_data["roads_coords"]

    tl_ids = tl_data["tl_ids"]
    tl_valid_split = tl_data["tl_valid"]
    tl_states_split = tl_data["tl_states"]

    XY = XY_split[:,ind_curr-n_channels+1:ind_curr+1]
    GT_XY = XY_split[:,ind_curr+1:]
    YAWS = YAWS_split[:,ind_curr-n_channels+1:ind_curr+1]
    agents_valid = agents_valid_split[:,ind_curr-n_channels+1:ind_curr+1]
    future_valid = agents_valid_split[:,ind_curr+1:]

    tl_valid_hist = tl_valid_split[:,ind_curr-n_channels+1:ind_curr+1]
    tl_states_hist = tl_states_split[:,ind_curr-n_channels+1:ind_curr+1]

    current_val = agents_valid[ag,-1]
    future_val = future_valid[ag]

    # print(XY.shape, GT_XY.shape)

    if (future_val.sum() == 0) or (current_val == 0):
        return None
        
    RES_ROADMAP = (
        np.ones((raster_size, raster_size, 3), dtype=np.uint8) * MAX_PIXEL_VALUE
    )
    RES_EGO = [
        np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
        for _ in range(n_channels)
    ]
    RES_OTHER = [
        np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
        for _ in range(n_channels)
    ]

    xy = XY[ag]
    yaw_ego = YAWS[ag]
    val = agents_valid[ag]
    ego_id = agents_ids[ag]
    gt_xy = GT_XY[ag]

    xy_val = xy[val > 0]
    if len(xy_val) == 0:
        return None


    unscaled_center_xy = xy_val[-1].reshape(1, -1)
    center_xy = unscaled_center_xy*zoom_fact
    yawt = yaw_ego[-1]
    rot_matrix = np.array(
        [
            [np.cos(yawt), -np.sin(yawt)],
            [np.sin(yawt), np.cos(yawt)],
        ]
    )

    centered_roadlines = (roads_coords - center_xy) @ rot_matrix + displacement
    centered_others = (XY.reshape(-1, 2)*zoom_fact - center_xy) @ rot_matrix + displacement
    centered_others = centered_others.reshape(XY.shape[0], n_channels, 2)
    centered_gt = (gt_xy - unscaled_center_xy) @ rot_matrix

    tl_dict = get_tl_dict(tl_states_hist[:,-1], tl_ids, tl_valid_hist[:,-1])

    RES_ROADMAP = draw_roads(RES_ROADMAP, centered_roadlines, roads_ids, tl_dict)



    # Agents

    

    unique_agent_ids = np.unique(agents_ids)

    

    is_ego = False
    # self_type = 0
    for other_agent_id in unique_agent_ids:
        other_agent_id = int(other_agent_id)
        if other_agent_id < 1:
            continue
        if other_agent_id == ego_id:
            is_ego = True
            # self_type = agent_type[agents_ids == other_agent_id]
        else:
            is_ego = False

        agent_lane = centered_others[agents_ids == other_agent_id][0]
        agent_valid = agents_valid[agents_ids == other_agent_id]
        agent_yaw = YAWS[agents_ids == other_agent_id]

        agent_l = lengths[agents_ids == other_agent_id]
        agent_w = widths[agents_ids == other_agent_id]

        for timestamp, (coord, valid_coordinate, past_yaw,) in enumerate(
            zip(
                agent_lane,
                agent_valid.flatten(),
                agent_yaw.flatten(),
            )
        ):
            if valid_coordinate == 0:
                continue
            box_points = (
                np.array(
                    [
                        -agent_l,
                        -agent_w,
                        agent_l,
                        -agent_w,
                        agent_l,
                        agent_w,
                        -agent_l,
                        agent_w,
                    ]
                )
                .reshape(4, 2)
                .astype(np.float32)
                *zoom_fact
                / 2
            )

            _coord = np.array([coord])
            
            yawt = yaw_ego[-1]            

            box_points = (
                box_points
                @ np.array(
                    (
                        (np.cos(yawt - past_yaw), -np.sin(yawt - past_yaw)),
                        (np.sin(yawt - past_yaw), np.cos(yawt - past_yaw)),
                    )
                ).reshape(2, 2)
            )

            box_points = box_points + _coord
            box_points = box_points.reshape(1, -1, 2).astype(np.int32)


            if is_ego:
                cv2.fillPoly(
                    RES_EGO[timestamp],
                    box_points,
                    color=MAX_PIXEL_VALUE
                )
            else:
                cv2.fillPoly(
                    RES_OTHER[timestamp],
                    box_points,
                    color=MAX_PIXEL_VALUE
                )

    raster = np.concatenate([RES_ROADMAP] + RES_EGO + RES_OTHER, axis=2)    

    raster_dict = {
        "object_id": ego_id,
        "raster": raster,
        "yaw": yaw_ego,
        "shift": unscaled_center_xy,
        "_gt_marginal": gt_xy,
        "gt_marginal": centered_gt,
        "future_val_marginal": future_val,
        # "gt_joint": GT_XY[tracks_to_predict.flatten() > 0],
        # "future_val_joint": future_valid[tracks_to_predict.flatten() > 0],
        # "scenario_id": scenario_id,
        # "self_type": self_type,
    }

    return raster_dict