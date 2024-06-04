import argparse
import multiprocessing
import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from feature_descriptor import features_description

fixed_frame = True
fixed_frame = False
use_rgb = True

MAX_PIXEL_VALUE = 255
N_ROADS = 21
road_colors = [int(x) for x in np.linspace(1, MAX_PIXEL_VALUE, N_ROADS).astype("uint8")]
idx2type = ["unset", "vehicle", "pedestrian", "cyclist", "other"]

state_to_color = {1:"red",4:"red",7:"red",2:"yellow",5:"yellow",8:"yellow",3:"green",6:"green"}
color_to_rgb = { "red":(255,0,0), "yellow":(255,255,0),"green":(0,255,0) }

egocol = {1:(0,128,0),0:(0,0,255)}

def draw_roads(roadmap, centered_roadlines, roadlines_ids, roadlines_types, tl_dict):

    unique_road_ids = np.unique(roadlines_ids)
    for road_id in unique_road_ids:
        if road_id >= 0:
            roadline = centered_roadlines[roadlines_ids == road_id]
            road_type = roadlines_types[roadlines_ids == road_id].flatten()[0]

            road_color = road_colors[road_type]
            for col in color_to_rgb.keys():
                if road_id in tl_dict[col]:
                    road_color = color_to_rgb[col]

            roadmap = cv2.polylines(
                roadmap,
                [roadline.astype(int)],
                False,
                road_color
            )

    return roadmap

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to raw data")
    parser.add_argument("--out", type=str, required=True, help="Path to save data")
    parser.add_argument(
        "--no-valid", action="store_true", help="Use data with flag `valid = 0`"
    )
    parser.add_argument(
        "--use-vectorize", action="store_true", help="Generate vector data"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=20, required=False, help="Number of threads"
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=8,
        required=False,
        help="Use `1/n_shards` of full dataset",
    )
    parser.add_argument(
        "--each",
        type=int,
        default=0,
        required=False,
        help="Take `each` sample in shard",
    )

    args = parser.parse_args()

    return args


def rasterize(parsed, validate):
    
    raster_size=224
    zoom_fact=3
    n_channels=11
    
    # Agents
    tracks_to_predict=parsed["state/tracks_to_predict"].numpy()
    past_x=parsed["state/past/x"].numpy()
    past_y=parsed["state/past/y"].numpy()
    current_x=parsed["state/current/x"].numpy()
    current_y=parsed["state/current/y"].numpy()
    current_yaw=parsed["state/current/bbox_yaw"].numpy()
    past_yaw=parsed["state/past/bbox_yaw"].numpy()
    past_valid=parsed["state/past/valid"].numpy()
    current_valid=parsed["state/current/valid"].numpy()
    agent_type=parsed["state/type"].numpy()
    future_x=parsed["state/future/x"].numpy()
    future_y=parsed["state/future/y"].numpy()
    future_valid=parsed["state/future/valid"].numpy()

    # Roads
    roadlines_coords=parsed["roadgraph_samples/xyz"].numpy()
    roadlines_types=parsed["roadgraph_samples/type"].numpy()
    roadlines_valid=parsed["roadgraph_samples/valid"].numpy()
    roadlines_ids=parsed["roadgraph_samples/id"].numpy()
    widths=parsed["state/current/width"].numpy()
    lengths=parsed["state/current/length"].numpy()
    agents_ids=parsed["state/id"].numpy()

    # Traffic lights
    tl_states=parsed["traffic_light_state/current/state"].numpy()
    tl_ids=parsed["traffic_light_state/current/id"].numpy()
    tl_valids=parsed["traffic_light_state/current/valid"].numpy()
    tl_past_states=parsed["traffic_light_state/past/state"].numpy()
    tl_past_valid=parsed["traffic_light_state/past/valid"].numpy()
    tl_current_x=parsed["traffic_light_state/current/x"].numpy()
    tl_current_y=parsed["traffic_light_state/current/y"].numpy()
    tl_past_x=parsed["traffic_light_state/past/x"].numpy()
    tl_past_y=parsed["traffic_light_state/past/y"].numpy()
    
    # Scenario
    scenario_id=parsed["scenario/id"].numpy()[0].decode("utf-8")

    
    # Prepare arrays
    GRES = []
    displacement = np.array([[raster_size // 2, raster_size // 2]])
    tl_dict = {"green": set(), "yellow": set(), "red": set()}

    # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
    # Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
    for tl_state, tl_id, tl_valid in zip(
        tl_states.flatten(), tl_ids.flatten(), tl_valids.flatten()
    ):
        if tl_valid == 0:
            continue
        if tl_state in [1, 4, 7]:
            tl_dict["red"].add(tl_id)
        if tl_state in [2, 5, 8]:
            tl_dict["yellow"].add(tl_id)
        if tl_state in [3, 6]:
            tl_dict["green"].add(tl_id)

    XY = np.concatenate(
        (
            np.expand_dims(np.concatenate((past_x, current_x), axis=1), axis=-1),
            np.expand_dims(np.concatenate((past_y, current_y), axis=1), axis=-1),
        ),
        axis=-1,
    )

    GT_XY = np.concatenate(
        (np.expand_dims(future_x, axis=-1), np.expand_dims(future_y, axis=-1)), axis=-1
    )

    YAWS = np.concatenate((past_yaw, current_yaw), axis=1)

    agents_valid = np.concatenate((past_valid, current_valid), axis=1)

    TL_current = np.transpose(np.concatenate((tl_current_x,tl_current_y),axis=0))
    tl_past_x, tl_current_x, tl_past_y, tl_current_y = np.transpose(tl_past_x), np.transpose(tl_current_x), np.transpose(tl_past_y), np.transpose(tl_current_y)
    TL_XY = np.concatenate(
        (
            np.expand_dims(np.concatenate((tl_past_x, tl_current_x), axis=1), axis=-1),
            np.expand_dims(np.concatenate((tl_past_y, tl_current_y), axis=1), axis=-1),
        ),
        axis=-1,
    )

    # Prepare roads
    roadlines_valid = roadlines_valid.reshape(-1)
    roadlines_coords = roadlines_coords[:, :2][roadlines_valid > 0]

    roadlines_types = roadlines_types[roadlines_valid > 0]
    roadlines_ids = roadlines_ids.reshape(-1)[roadlines_valid > 0]

    # Center in map, useful for offline rendering
    mapcenter = roadlines_coords.mean(0)
    roadlines_coords = roadlines_coords - mapcenter
    XY = XY - mapcenter
    GT_XY = GT_XY - mapcenter
    TL_XY = TL_XY - mapcenter

    roadlines_coords = roadlines_coords*zoom_fact

    # Loop over agents to track
    for _, (
        xy,
        current_val,
        val,
        ag_type,
        yaw,
        agent_id,
        gt_xy,
        future_val,
        predict,
        yawvec
    ) in enumerate(
        zip(
            XY,
            current_valid,
            agents_valid,
            agent_type,
            current_yaw.flatten(),
            agents_ids,
            GT_XY,
            future_valid,
            tracks_to_predict.flatten(),
            YAWS
        )
    ):
        if (not validate and future_val.sum() == 0) or (validate and predict == 0):
            continue
        if current_val == 0:
            continue
        # Apply only to vehicles
        if ag_type!=1:
            continue


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
        RES_ROADSTIME = [
            np.ones((raster_size, raster_size, 3), dtype=np.uint8) * MAX_PIXEL_VALUE
            for _ in range(n_channels)
        ]

        xy_val = xy[val > 0]
        if len(xy_val) == 0:
            continue

        # pab
        # TL_current = TL_current*zoom_fact
        # TL_current = (TL_current - center_xy) @ rot_matrix + displacement

        # tl_valids = tl_valids.reshape(-1)
        # tl_states = tl_states.reshape(-1)

        # # Show traffic lights
        # for j, tl in enumerate(tl_states):
        #     if tl_valids[j]<=0 or tl==0:
        #         continue

        #     col_tl = color_to_rgb[state_to_color[tl]]
                
        #     RES_ROADMAP = cv2.circle(
        #         RES_ROADMAP,
        #         tuple(TL_current[j].astype(np.int32)),
        #         radius=5,
        #         color=col_tl,
        #         thickness=-1
        #     )
        

        if fixed_frame:

            unscaled_center_xy = xy_val[-1].reshape(1, -1)
            center_xy = unscaled_center_xy*zoom_fact
            rot_matrix = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)],
                ]
            )

            centered_roadlines = (roadlines_coords - center_xy) @ rot_matrix + displacement
            centered_others = (XY.reshape(-1, 2)*zoom_fact - center_xy) @ rot_matrix + displacement
            centered_others = centered_others.reshape(128, n_channels, 2)
            centered_gt = (gt_xy - unscaled_center_xy) @ rot_matrix

            RES_ROADMAP = draw_roads(RES_ROADMAP, centered_roadlines, roadlines_ids, roadlines_types, tl_dict)

        else:

            if xy_val.shape[0]<n_channels:
                continue

            for timestamp in range(n_channels):

                unscaled_center_xy = xy_val[timestamp].reshape(1, -1)
                center_xy = unscaled_center_xy*zoom_fact
                yawt = yawvec[timestamp]#YAWS[ind,timestamp]
                rot_matrix = np.array(
                    [
                        [np.cos(yawt), -np.sin(yawt)],
                        [np.sin(yawt), np.cos(yawt)],
                    ]
                )

                centered_roadlines = (roadlines_coords - center_xy) @ rot_matrix + displacement
                centered_others = XY
                centered_gt = (gt_xy - unscaled_center_xy) @ rot_matrix

                RES_ROADSTIME[timestamp] = draw_roads(RES_ROADSTIME[timestamp], centered_roadlines, roadlines_ids, roadlines_types, tl_dict)

        # Agents

        unique_agent_ids = np.unique(agents_ids)

        is_ego = False
        self_type = 0
        _tmp = 0
        for other_agent_id in unique_agent_ids:
            other_agent_id = int(other_agent_id)
            if other_agent_id < 1:
                continue
            if other_agent_id == agent_id:
                is_ego = True
                self_type = agent_type[agents_ids == other_agent_id]
            else:
                is_ego = False

            _tmp += 1
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
                
                if  fixed_frame:
                    yawt = yaw
                else:
                    yawt = yawvec[timestamp]#YAWS[ind,timestamp]
                    unscaled_center_xy = xy_val[timestamp].reshape(1, -1)
                    center_xy = unscaled_center_xy*zoom_fact
                    _coord = (_coord.reshape(-1, 2)*zoom_fact - center_xy) @ rot_matrix + displacement

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

                if not fixed_frame and use_rgb:
                    RES_ROADSTIME[timestamp] = cv2.fillPoly(
                            RES_ROADSTIME[timestamp],
                            box_points,
                            color=egocol[is_ego]
                        )
                else:
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

        if fixed_frame:
            raster = np.concatenate([RES_ROADMAP] + RES_EGO + RES_OTHER, axis=2)
        else:
            timarray = np.array(RES_ROADSTIME)
            
            if use_rgb:
                raster = timarray
            else:   
                egoarray, otherarray = np.array(RES_EGO), np.array(RES_OTHER)
                roads_onechannel = timarray.mean(axis=-1)#,keepdims=True)
                raster = roads_onechannel + egoarray.squeeze()/2. + otherarray.squeeze()
                raster = raster.transpose(1, 2, 0)

        raster_dict = {
            "object_id": agent_id,
            "raster": raster,
            "yaw": yaw,
            "shift": unscaled_center_xy,
            "_gt_marginal": gt_xy,
            "gt_marginal": centered_gt,
            "future_val_marginal": future_val,
            "gt_joint": GT_XY[tracks_to_predict.flatten() > 0],
            "future_val_joint": future_valid[tracks_to_predict.flatten() > 0],
            "scenario_id": scenario_id,
            "self_type": self_type,
        }

        GRES.append(raster_dict)

    return GRES


F2I = {
    "x": 0,
    "y": 1,
    "s": 2,
    "vel_yaw": 3,
    "bbox_yaw": 4,
    "l": 5,
    "w": 6,
    "agent_type_range": [7, 12],
    "lane_range": [13, 33],
    "lt_range": [34, 43],
    "global_idx": 44,
}


def ohe(N, n, zero):
    n = int(n)
    N = int(N)
    M = np.eye(N)
    diff = 0
    if zero:
        M = np.concatenate((np.zeros((1, N)), M), axis=0)
        diff = 1
    return M[n + diff]


def make_2d(arraylist):
    n = len(arraylist)
    k = arraylist[0].shape[0]
    a2d = np.zeros((n, k))
    for i in range(n):
        a2d[i] = arraylist[i]
    return a2d


def vectorize(
    past_x,
    current_x,
    past_y,
    current_y,
    past_valid,
    current_valid,
    past_speed,
    current_speed,
    past_velocity_yaw,
    current_velocity_yaw,
    past_bbox_yaw,
    current_bbox_yaw,
    Agent_id,
    Agent_type,
    Roadline_id,
    Roadline_type,
    Roadline_valid,
    Roadline_xy,
    Tl_rl_id,
    Tl_state,
    Tl_valid,
    W,
    L,
    tracks_to_predict,
    future_x,
    future_y,
    future_valid,
    validate,
    n_channels=11,
):

    XY = np.concatenate(
        (
            np.expand_dims(np.concatenate((past_x, current_x), axis=1), axis=-1),
            np.expand_dims(np.concatenate((past_y, current_y), axis=1), axis=-1),
        ),
        axis=-1,
    )

    Roadline_valid = Roadline_valid.flatten()
    RoadXY = Roadline_xy[:, :2][Roadline_valid > 0]
    Roadline_type = Roadline_type[Roadline_valid > 0].flatten()
    Roadline_id = Roadline_id[Roadline_valid > 0].flatten()

    tl_state = [[-1] for _ in range(9)]

    for lane_id, state, valid in zip(
        Tl_rl_id.flatten(), Tl_state.flatten(), Tl_valid.flatten()
    ):
        if valid == 0:
            continue
        tl_state[int(state)].append(lane_id)

    VALID = np.concatenate((past_valid, current_valid), axis=1)

    Speed = np.concatenate((past_speed, current_speed), axis=1)
    Vyaw = np.concatenate((past_velocity_yaw, current_velocity_yaw), axis=1)
    Bbox_yaw = np.concatenate((past_bbox_yaw, current_bbox_yaw), axis=1)

    GRES = []

    ROADLINES_STATE = []

    GLOBAL_IDX = -1

    GT_XY = np.concatenate(
        (np.expand_dims(future_x, axis=-1), np.expand_dims(future_y, axis=-1)), axis=-1
    )

    real_valid = future_valid.sum(1)>0
    gt_arr = GT_XY[real_valid]
    val_arr = future_valid[real_valid]

    unique_road_ids = np.unique(Roadline_id)
    for road_id in unique_road_ids:

        GLOBAL_IDX += 1

        roadline_coords = RoadXY[Roadline_id == road_id]
        roadline_type = Roadline_type[Roadline_id == road_id][0]

        for i, (x, y) in enumerate(roadline_coords):
            if i > 0 and i < len(roadline_coords) - 1 and i % 3 > 0:
                continue
            tmp = np.zeros(48)
            tmp[0] = x
            tmp[1] = y

            tmp[13:33] = ohe(20, roadline_type, True)

            tmp[44] = GLOBAL_IDX

            ROADLINES_STATE.append(tmp)

    ROADLINES_STATE = make_2d(ROADLINES_STATE)

    for (
        agent_id,
        xy,
        current_val,
        valid,
        _,
        bbox_yaw,
        _,
        _,
        _,
        future_val,
        predict,
    ) in zip(
        Agent_id,
        XY,
        current_valid,
        VALID,
        Speed,
        Bbox_yaw,
        Vyaw,
        W,
        L,
        future_valid,
        tracks_to_predict.flatten(),
    ):

        if (not validate and future_val.sum() == 0) or (validate and predict == 0):
            continue
        if current_val == 0:
            continue

        GLOBAL_IDX = -1
        RES = []

        xy_val = xy[valid > 0]
        if len(xy_val) == 0:
            continue

        # pab
        #centered_xy = xy_val[-1].copy().reshape(-1, 2)
        centered_xy = ROADLINES_STATE[:, :2].mean(0).reshape(-1, 2)

        ANGLE = bbox_yaw[-1]

        # rot_matrix = np.array(
        #     [
        #         [np.cos(ANGLE), -np.sin(ANGLE)],
        #         [np.sin(ANGLE), np.cos(ANGLE)],
        #     ]
        # ).reshape(2, 2)
        rot_matrix = np.array(
            [
                [1., 0.],
                [0., 1.],
            ]
        ).reshape(2, 2)

        local_roadlines_state = ROADLINES_STATE.copy()

        local_roadlines_state[:, :2] = (
            local_roadlines_state[:, :2] - centered_xy
        ) @ rot_matrix.astype(np.float64)

        local_XY = ((XY - centered_xy).reshape(-1, 2) @ rot_matrix).reshape(
            128, n_channels, 2
        )

        for (
            other_agent_id,
            other_agent_type,
            other_xy,
            other_valids,
            other_speeds,
            other_bbox_yaws,
            other_v_yaws,
            other_w,
            other_l,
            other_predict,
        ) in zip(
            Agent_id,
            Agent_type,
            local_XY,
            VALID,
            Speed,
            Bbox_yaw,
            Vyaw,
            W.flatten(),
            L.flatten(),
            tracks_to_predict.flatten(),
        ):
            if other_valids.sum() == 0:
                continue

            GLOBAL_IDX += 1
            for timestamp, (
                (x, y),
                v,
                other_speed,
                other_v_yaw,
                other_bbox_yaw,
            ) in enumerate(
                zip(other_xy, other_valids, other_speeds, other_v_yaws, other_bbox_yaws)
            ):
                if v == 0:
                    continue
                tmp = np.zeros(48)
                tmp[0] = x
                tmp[1] = y
                tmp[2] = other_speed
                tmp[3] = other_v_yaw - ANGLE
                tmp[4] = other_bbox_yaw - ANGLE
                tmp[5] = float(other_l)
                tmp[6] = float(other_w)

                tmp[7:12] = ohe(5, other_agent_type, True)

                tmp[43] = timestamp

                tmp[44] = GLOBAL_IDX
                tmp[45] = 1 if other_agent_id == agent_id else 0
                tmp[46] = other_predict
                tmp[47] = other_agent_id

                RES.append(tmp)
        local_roadlines_state[:, 44] = local_roadlines_state[:, 44] + GLOBAL_IDX + 1
        RES = np.concatenate((make_2d(RES), local_roadlines_state), axis=0)
        GRES.append(RES)

    return GRES, centered_xy, gt_arr, val_arr


def merge(data, proc_id, validate, out_dir, use_vectorize=False, max_rand_int=10000000000):

    parsed = tf.io.parse_single_example(data, features_description)
    raster_data = rasterize(parsed,validate=validate)

    if use_vectorize:
        vector_data, center, gtall, valall = vectorize(
            parsed["state/past/x"].numpy(),
            parsed["state/current/x"].numpy(),
            parsed["state/past/y"].numpy(),
            parsed["state/current/y"].numpy(),
            parsed["state/past/valid"].numpy(),
            parsed["state/current/valid"].numpy(),
            parsed["state/past/speed"].numpy(),
            parsed["state/current/speed"].numpy(),
            parsed["state/past/vel_yaw"].numpy(),
            parsed["state/current/vel_yaw"].numpy(),
            parsed["state/past/bbox_yaw"].numpy(),
            parsed["state/current/bbox_yaw"].numpy(),
            parsed["state/id"].numpy(),
            parsed["state/type"].numpy(),
            parsed["roadgraph_samples/id"].numpy(),
            parsed["roadgraph_samples/type"].numpy(),
            parsed["roadgraph_samples/valid"].numpy(),
            parsed["roadgraph_samples/xyz"].numpy(),
            parsed["traffic_light_state/current/id"].numpy(),
            parsed["traffic_light_state/current/state"].numpy(),
            parsed["traffic_light_state/current/valid"].numpy(),
            parsed["state/current/width"].numpy(),
            parsed["state/current/length"].numpy(),
            parsed["state/tracks_to_predict"].numpy(),
            parsed["state/future/x"].numpy(),
            parsed["state/future/y"].numpy(),
            parsed["state/future/valid"].numpy(),
            validate=validate,
        )


    for i in range(len(raster_data)):
        if use_vectorize:
            raster_data[i]["vector_data"] = vector_data[i].astype(np.float16)
            raster_data[i]["center"] = center
            raster_data[i]["gt_all"] = gtall
            raster_data[i]["val_all"] = valall

        r = np.random.randint(max_rand_int)
        filename = f"{idx2type[int(raster_data[i]['self_type'])]}_{proc_id}_{str(i).zfill(5)}_{r}.npz"
        np.savez_compressed(os.path.join(out_dir, filename), **raster_data[i])


def main():
    args = parse_arguments()
    print(args)

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    files = os.listdir(args.data)
    dataset = tf.data.TFRecordDataset(
        [os.path.join(args.data, f) for f in files], num_parallel_reads=1
    )
    if args.n_shards > 1:
        dataset = dataset.shard(args.n_shards, args.each)

    p = multiprocessing.Pool(args.n_jobs)
    proc_id = 0
    res = []
    for data in tqdm(dataset.as_numpy_iterator()):
        proc_id += 1
        res.append(
            p.apply_async(
                merge,
                kwds=dict(
                    data=data,
                    proc_id=proc_id,
                    validate=not args.no_valid,
                    out_dir=args.out,
                    use_vectorize=args.use_vectorize,
                ),
            )
        )

    for r in tqdm(res):
        r.get()


if __name__ == "__main__":
    main()