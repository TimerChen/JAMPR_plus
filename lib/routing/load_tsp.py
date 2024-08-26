import pandas as pd
import pickle as pkl
from lib.routing.formats import RPInstance
import numpy as np
import torch
from random import choice

from lib.routing.env import RPEnv
from lib.utils.challenge_utils import dimacs_challenge_dist_fn

def valid_instance(coords: torch.Tensor, demand: torch.Tensor, tw: torch.Tensor, service_time: torch.Tensor, org_service_horizon: float) -> bool:
    dist_to_depot = dimacs_challenge_dist_fn(coords[1:], coords[0])
    time_to_depot = dist_to_depot / org_service_horizon

    return not (tw[1:, 1] + time_to_depot + service_time > 1.).any()

def load_tsplib_instance(pth: str):
    """ Load just one randomly-chosen instance from jxchen's tsptw daset in JAMPR's format """
    with open(pth, "rb") as fd:
        dataset = pkl.load(fd)

    i = choice(range(len(dataset["data"])))
    coord = dataset["data"][i]
    tw = dataset["tw"][i]

    instance = {"max_vehicle_number": 100, "vehicle_capacity": 100}
    features = []
    feature_names = ["node_id", "x_coord", "y_coord", "demand", "tw_start", "tw_end", "service_time"]

    for j in range(coord.shape[0]):
        features.append([j, coord[j, 0], coord[j, 1], 1.0 if j > 0 else 0.0, tw[j, 0], tw[j, 1], 0.0])

    df = pd.DataFrame(data=features, columns=feature_names)
    df.set_index("node_id")
    df.drop(labels="node_id", axis=1, inplace=True)
    df["tw_len"] = df.tw_end - df.tw_start

    instance["features"] = df

    return instance

def load_tsptw_instances(pths) -> RPInstance:
    if isinstance(pths, str):
        pths = [pths]
    ret = []
    for p in pths:
        ret.extend(load_tsptw_instances_one(p))
    return ret

def load_tsptw_instances_one(pth: str) -> RPInstance:
    """ Load an instance from jxchen's tsptw format
    """
    with open(pth, "rb") as f:
        dataset = pkl.load(f)
    rets = []

    tf = torch.tensor(dataset["time_factor"])
    lf = torch.tensor(dataset["loc_factor"])
    data = dataset['data']
    tws = dataset["tw"]

    # forcely remove tw
    no_tw = False

    # edge = dataset['edges']
    # soln = dataset['seq']
    # val = dataset['val']

    # lf = lf[offset:offset + num_samples]
    # self.tf = tf / lf
    # self.data = [torch.FloatTensor(row) / lf[i] for i, row in enumerate(data[offset:offset + num_samples])]
    # self.edge = [torch.FloatTensor(row) / lf[i] for i, row in enumerate(edge[offset:offset + num_samples])]
    # self.tw = [torch.FloatTensor(row) / (lf[i]) for i, row in enumerate(tw[offset:offset + num_samples])]
    # self.soln = [torch.LongTensor(row) for row in (soln[offset:offset + num_samples])]
    # self.opt_val = torch.FloatTensor(val[offset:offset + num_samples]) / lf
    # self.time = torch.FloatTensor(time[offset:offset + num_samples])

    gsize = len(data[0])
    for i in range(len(dataset["data"])):
        if len(dataset['seq'][i]) == 0:
            continue

        org_service_horizon = tws[i][0, 1]
        coords = torch.tensor(data[i]) / lf[i]
        tw = torch.tensor(tws[i]) / org_service_horizon
        if no_tw:
            tw[:] = tw[0]
        demand = torch.tensor([0.01] * gsize)
        service_time = torch.tensor(0)

        # assert valid_instance(coords, demand, tw, service_time, org_service_horizon)
        
        ret = RPInstance(
            coords=coords,
            demands=demand,
            tw=tw,
            service_time=service_time,
            graph_size=coords.shape[0],
            org_service_horizon=org_service_horizon,
            max_vehicle_number=100,
            vehicle_capacity=1.0,  # is normalized
            service_horizon=1.0,  # is normalized
            depot_idx=[0],
            type="1",
            tw_frac=str("1"), # all nodes have time-window constrain.
            expert_sol=dataset['seq'][i]
        )
        rets.append(ret)

    return rets

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from lib.routing import RPDataset, GROUPS, TYPES, TW_FRACS

    SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
    SMP = 128
    BS = 64
    BS_ = BS
    N = 21
    DPATH = "./21_train.pkl"
    LPATH = "./solomon_stats.pkl"
    CUDA = False
    MAX_CON = 1
    SEED = 0
    device = torch.device("cuda" if CUDA else "cpu")
    if True:
        ds = RPDataset(cfg=SAMPLE_CFG, data_pth=DPATH)
    else:
        ds = RPDataset(cfg=SAMPLE_CFG, stats_pth=LPATH)
    data = ds.sample(sample_size=SMP, graph_size=N, custom_func=load_tsptw_instances)
    # print(data[-1].tw, data[-1].coords)

    dl = DataLoader(
        data,
        batch_size=BS_,
        collate_fn=lambda x: x,  # identity -> returning simple list of instances
        shuffle=False
    )

    env = RPEnv(
                debug=False,
                device=device,
                max_concurrent_vehicles=1,
                k_nbh_frac=1,
                pomo=False,
                pomo_single_start_node=True,
                num_samples=1,
                inference=False,
                late_penalty=10.,
                late_penalty_factor=10.
                )
    env.seed(SEED+1)
    see_idx = 9

    for batch in dl:
        print(batch[see_idx].tw)
        env.load_data(batch)
        obs = env.reset()
        done = False
        i = 0
        start_tws = env._stack_to_tensor(batch, "tw")[:, :, 1]

        while not done:
            tr = torch.randint(MAX_CON, size=(BS,), device=device)
            t_nbh = obs.nbh[torch.arange(BS), tr]
            t_msk = obs.nbh_mask[torch.arange(BS), tr]
            # if i == 0:
            print(t_nbh[see_idx], t_msk[see_idx])
            nd = torch.zeros(BS, dtype=torch.long, device=device)
            for j, (nbh, msk, start_tw) in enumerate(zip(t_nbh, t_msk, start_tws)):
                available_idx = nbh[~msk]   # mask is True where infeasible
                idx = available_idx[start_tw[available_idx].argsort(-1, descending=False)]
                if j == see_idx:
                    print("ava idx", idx, start_tw[0])
                nd[j] = idx[0]
            print("current_time", env.cur_time.shape, env.cur_time[see_idx])
            
            obs, rew, done, info = env.step(torch.stack((tr, nd), dim=-1))
            print(nd[see_idx], "rew", rew[see_idx])
            print("late", env.exceeds_tw[see_idx])

            i += 1

        sol = env.export_sol()
        # print(type(sol), len(sol), len(sol[0]), len(sol[1]))
        print(sol[0][see_idx], sol[1][see_idx])
        print((np.array(sol[0])==0).any())

        break