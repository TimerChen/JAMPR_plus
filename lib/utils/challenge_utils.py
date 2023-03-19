#
from typing import Union, List, Tuple
import numpy as np
import torch


def dimacs_challenge_dist_fn_np(i: Union[np.ndarray, float],
                                j: Union[np.ndarray, float],
                                scale: int = 100,
                                ) -> np.ndarray:
    """
    times/distances are obtained from the location coordinates,
    by computing the Euclidean distances truncated to one
    decimal place:
    $d_{ij} = \frac{\floor{10e_{ij}}}{10}$
    where $e_{ij}$ is the Euclidean distance between locations i and j

    coords*100 since they were normalized to [0, 1]
    """
    return np.floor(10*np.sqrt(((scale*(i - j))**2).sum(axis=-1)))/10


@torch.jit.script
def dimacs_challenge_dist_fn(i: torch.Tensor, j: torch.Tensor, scale: int = 100,) -> torch.Tensor:
    """
    times/distances are obtained from the location coordinates,
    by computing the Euclidean distances truncated to one
    decimal place:
    $d_{ij} = \frac{\floor{10e_{ij}}}{10}$
    where $e_{ij}$ is the Euclidean distance between locations i and j

    coords*100 since they were normalized to [0, 1]
    """
    return torch.floor(10*torch.sqrt(((scale*(i - j))**2).sum(dim=-1)))/10

def eval_tsp_sols(sols: List[List[int]], expert_sols: List[List[int]], coords: List[torch.Tensor], tws: List[torch.Tensor], org_service_horizons: List[Union[float, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if_valid = []
    if_exp_valid = []
    gaps = []

    for i, sol in enumerate(sols):
        expert_sol = expert_sols[i]
        if sol[0] == 0:
            sol = sol[1:]
        if sol[-1] != 0:
            sol.append(0)
        if expert_sol[0] == 0:
            expert_sol = expert_sol[1:]
        if expert_sol[-1] != 0:
            expert_sol.append(0)

        time_mat = dimacs_challenge_dist_fn(coords[i][:, None, :], coords[i][None, :, :]) / org_service_horizons[i]

        prev = 0
        exp_prev = 0
        tm = torch.tensor(0.)
        exp_tm = torch.tensor(0.)
        valid = True
        exp_valid = True
        for j, nxt in enumerate(sol):
            exp_nxt = expert_sol[j]

            tm += time_mat[prev, nxt]
            exp_tm += time_mat[exp_prev, exp_nxt]

            if tm > tws[i][nxt, 1]:
                valid = False
            if exp_tm > tws[i][exp_nxt, 1]:
                exp_valid = False

            tm += (tws[i][nxt, 0] - tm).clamp_(min=0)
            exp_tm += (tws[i][exp_nxt, 0] - exp_tm).clamp_(min=0)

            prev = nxt
            exp_prev = exp_nxt
        
        if_valid.append(valid)
        if_exp_valid.append(exp_valid)
        gaps.append((tm - exp_tm).item())

    return torch.tensor(if_valid), torch.tensor(if_exp_valid), torch.tensor(gaps)


# ============= #
# ### TEST #### #
# ============= #
def _test():
    rnd = np.random.default_rng(1)
    np_coords = rnd.uniform(0, 1, size=20).reshape(-1, 2)
    np_dists = dimacs_challenge_dist_fn_np(np_coords[1:], np_coords[0])
    pt_coords = torch.from_numpy(np_coords)
    pt_dists = dimacs_challenge_dist_fn(pt_coords[1:], pt_coords[0])
    assert np.all(np_dists == pt_dists.cpu().numpy())
