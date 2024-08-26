#
import time
import math
import logging
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Tuple, Any
from torch import Tensor
from omegaconf import DictConfig

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from lib.routing import RPEnv, RPInstance, RPDataset
from lib.utils.runner_utils import MonitorCallback, CheckpointCallback
from lib.utils.challenge_utils import eval_tsp_sols
from lib.model.policy import Policy
from lib.model.baselines.base_class import Baseline

logger = logging.getLogger(__name__)


def train_episode(data: List[RPInstance],
                  env: RPEnv,
                  policy: Policy,
                  **kwargs) -> Tuple[Tensor, Tensor, Union[Tensor, Any], Union[List, Dict]]:
    """Do one episode of the environment."""

    policy.train()
    policy.set_decode_type('sampling')
    policy.reset_static()

    env.load_data(data)
    obs = env.reset()

    costs, logs, entropy = [], [], []
    info = {}
    done = False

    actions = []
    # policy is <class 'lib.model.policy.Policy'>
    # print("policy is", type(policy))
    while not done:
        action, log_likelihood, ent = policy(obs)
        # print("obs ", obs.nbh_mask.shape, obs.nbh[0], obs.nbh_mask[0], action[0])
        obs, cost, done, info = env.step(action)
        actions.append(action)

        costs.append(cost)
        logs.append(log_likelihood)
        if ent is not None:
            entropy.append(ent)

    # acts = torch.stack(actions, dim=0)
    # print("act: ", acts.shape, acts[:, 0])

    # costs = sum(costs)
    for i in range(len(costs)-2, -1, -1):
        costs[i] = costs[i] + costs[i+1] * 0.98
    costs = torch.stack(costs, dim=1)
    logs = torch.stack(logs, dim=1)
    entropy = torch.stack(entropy, dim=0).mean() if entropy else None
    return costs, logs, entropy, info


def eval_episode(data: List[RPInstance],
                 env: RPEnv,
                 policy: Policy,
                 render: bool = False,
                 sampling: bool = False,
                 **kwargs) -> Tuple[Tensor, Union[List, Dict]]:
    """Do one episode of the environment."""

    policy.eval()
    policy.set_decode_type('sampling' if sampling and not env.pomo else 'greedy')
    policy.reset_static()

    bs = len(data)

    env.load_data(data)
    obs = env.reset()

    costs = []
    actions = []
    info = None
    done = False
    while not done:
        action, _, _ = policy(obs)
        obs, cost, done, info = env.step(action)
        if render:
            env.render(**kwargs)
        costs.append(cost)
        actions.append(action)

    # print("eval cost", len(costs), costs[0].shape, costs[-1].shape)
    costs = sum(costs).cpu()
    sols = [[a[i][1].item() for a in actions] for i in range(bs)]
    tps = list(map(lambda i: (i.expert_sol, i.coords, i.tw, i.org_service_horizon), data))
    if_valid, exp_valid, gaps, tm_gaps = eval_tsp_sols(sols, *map(lambda x: list(x), zip(*tps)))

    info["valid_rate"] = if_valid.sum() / if_valid.shape[0]
    info["expert_valid_rate"] = exp_valid.sum() / exp_valid.shape[0]
    info["avg_path_gap"] = gaps.mean()
    info["avg_valid_path_gap"] = gaps[if_valid].mean()
    info["avg_tm_gap"] = tm_gaps[if_valid].mean()

    if env.num_samples > 1:
        # select best sample
        c = info['current_total_cost']
        # select best sample
        c = c.reshape(-1, env.num_samples)
        best_idx = c.argmin(axis=-1)
        idx_range = np.arange(c.shape[0])
        info = {
            k: (
                v.reshape(-1, env.num_samples)[idx_range, best_idx]
                if isinstance(v, np.ndarray) else v
            )
            for k, v in info.items()
        }
        costs = costs.view(-1, env.num_samples)[idx_range, best_idx]

    return costs, info


def rollout(dataset: RPDataset,
            env: RPEnv,
            policy: Policy,
            batch_size: int,
            num_workers: int = 4,
            disable_progress_bar: bool = False,
            **kwargs) -> Tuple[Tensor, Union[List, Dict]]:
    """Policy evaluation rollout

    Args:
        dataset: dataset to evaluate on
        env: the routing simulation environment
        policy: policy model
        batch_size: size of mini-batches
        num_workers: num cores to distribute data loading
        disable_progress_bar: flag to disable tqdm progress bar

    Returns:
        tensor of final costs per instance

    """

    costs, infos = [], []
    for batch in tqdm(
        DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda x: x,  # identity -> returning simple list of instances
            shuffle=False   # do not random shuffle data in rollout!
        ),
        disable=disable_progress_bar,
    ):
        with torch.no_grad():
            cost, info = eval_episode(batch, env, policy, **kwargs)
        costs.append(cost.cpu())
        infos.append(info)

    env.clear_cache()
    return torch.cat(costs, dim=0), infos


def validate(
        dataset: RPDataset,
        env: RPEnv,
        policy: Policy,
        cfg: Optional[Dict] = None,
        **kwargs) -> Dict:
    """

    Args:
        dataset: dataset to evaluate on
        env: the routing simulation environment
        policy: policy model
        cfg: additional configuration

    Returns:
        mean of cost over all instances in dataset
    """
    cfg = cfg if cfg is not None else {}
    cost, infos = rollout(dataset, env, policy, **cfg, **kwargs)
    k_used = np.concatenate([np.array(i['k_used']).reshape(-1) for i in infos])
    costs = np.concatenate([i['current_total_cost'] for i in infos], axis=-1)

    late_rate = np.concatenate([np.array(i['late_rate']).reshape(-1) for i in infos])
    late_num = np.concatenate([np.array(i['late_num']).reshape(-1) for i in infos])
    late_cost = np.concatenate([np.array(i['late_cost']).reshape(-1) for i in infos])
    late_time = np.concatenate([np.array(i['late_time']).reshape(-1) for i in infos])

    valid_rate = np.concatenate([np.array(i["valid_rate"]).reshape(-1) for i in infos])
    expert_valid_rate = np.concatenate([np.array(i["expert_valid_rate"]).reshape(-1) for i in infos])
    avg_path_gap = np.concatenate([np.array(i["avg_path_gap"]).reshape(-1) for i in infos])
    avg_valid_path_gap = np.concatenate([np.array(i["avg_valid_path_gap"]).reshape(-1) for i in infos])
    avg_tm_gap = np.concatenate([np.array(i["avg_tm_gap"]).reshape(-1) for i in infos])

    return {
        "cost": cost.mean().item(),
        "cost_std": cost.std().item(),
        "k_used": np.mean(k_used),
        "k_used_std": np.std(k_used),
        "k_used_max": np.max(k_used),
        # "final_costs": costs,
        "final_costs_mean": costs.mean(),
        "late_rate": late_rate.mean().item(),
        "late_num": late_num.mean().item(),
        "late_cost": late_cost.mean().item(),
        "late_time": late_time.mean().item(),
        "valid_rate": valid_rate.mean().item(),
        "expert_valid_rate": expert_valid_rate.mean().item(),
        "avg_path_gap": avg_path_gap[avg_path_gap == avg_path_gap].mean().item(),
        "avg_valid_path_gap": avg_valid_path_gap[avg_valid_path_gap == avg_valid_path_gap].mean().item(),
        "avg_tm_gap": avg_tm_gap[avg_tm_gap == avg_tm_gap].mean().item(),
        "dataset": dataset.data_pth
    }


def train_batch(batch: List,
                env: RPEnv,
                policy: Policy,
                optimizer: torch.optim.Optimizer,
                baseline: Baseline,
                entropy_coeff: float = 0.0,
                bl_coeff: float = 0.8,
                max_grad_norm: float = math.pi
                ):
    """Do a full batched training episode and update model."""

    # get baseline values
    batch, bl_val = baseline.unwrap_batch(batch)

    # execute model, get costs and log probabilities
    cost, log_p, entropy, infos = train_episode(batch, env, policy)

    # Evaluate baseline and get loss (e.g. when using critic)
    bl_loss = 0
    if bl_val is None:
        bl_val, bl_loss = baseline.eval(batch, cost)

    # Calculate loss
    # print("t: shapes", cost.shape, bl_val.shape, log_p.shape, len(batch))
    pg_loss = ((cost[:, :, None] - bl_val) * log_p).mean()
    # pg_loss = -((cost - bl_val) * log_p).mean()
    # print("t: cost{:.3f}, bl_val{:.3f}, adv{:.3f}".format(cost.mean(), bl_val.mean(), (cost-bl_val).mean()))

    # add entropy regularization (pos coefficient) / bonus (neg coefficient)
    ent = entropy_coeff*entropy if entropy is not None else 0

    # final loss to be minimized
    loss = pg_loss + bl_coeff*bl_loss + ent
    print("loss: {:.3f}, pg_loss: {:.3f}, bl_loss: {:.3f}, ent: {:.3f}".format(loss, pg_loss, bl_loss, ent))
    # backward, gradient clipping and weight update
    optimizer.zero_grad()
    loss.backward()
    grad_norm = clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    # format result log
    k_used = infos['k_used']
    return {
        "cost": cost.mean().cpu().item(),
        "cost_std": cost.std().cpu().item(),
        "k_used": np.mean(k_used),
        "k_used_std": np.std(k_used),
        "pg_loss": pg_loss.cpu().item(),
        "critic_loss": bl_loss if bl_loss > 0 else 0,
        "entropy": entropy.cpu().item() if entropy is not None else 0,
        "grad_norm": grad_norm.cpu().item(),
    }


def train(
        cfg: Union[Dict, DictConfig],
        train_dataset: RPDataset,
        val_dataset: RPDataset,
        train_env: RPEnv,
        val_env: RPEnv,
        policy: Policy,
        optimizer: torch.optim.Optimizer,
        baseline: Baseline,
        lr_scheduler,
        monitor: MonitorCallback,
        ckpt_cb: CheckpointCallback,
        resume: bool = False,
        verbose: int = 2,
        render_val: bool = False,
        **kwargs) -> Dict:
    """Training function executing whole training procedure.

    Args:
        cfg:
        train_dataset:
        val_dataset:
        train_env:
        val_env:
        policy:
        optimizer:
        baseline:
        lr_scheduler:
        monitor:
        ckpt_cb:
        resume:
        verbose:
        render_val:
        **kwargs:

    Returns:

    """

    no_p_bar = cfg['disable_progress_bar'] or verbose == 0
    num_epochs = cfg['num_epochs']
    start_epoch, epoch, n_episodes, gradient_step = 0, 0, 0, 0
    t_start = time.time()
    t_sample, t_baseline, t_epoch = 0, 0, 0

    if resume:
        start_epoch, n_episodes, gradient_step = monitor.restore_data()
    if val_dataset.data is None:
        val_dataset.sample(sample_size=cfg['val_dataset_size'], graph_size=cfg['graph_size'])

    val_result = validate(
        val_dataset, val_env, policy,
        batch_size=cfg['val_batch_size'],
        num_workers=cfg['num_workers'],
        disable_progress_bar=no_p_bar,
        render=render_val,

    )
    monitor.log_eval_data(val_result, n_episodes, mode="val")

    best_epoch = start_epoch
    best_cost, best_cost_std = val_result['cost'], val_result['cost_std']
    late_rate = val_result['late_rate']
    late_cost = val_result["late_cost"]
    late_num = val_result["late_num"]

    # execute 'num_epochs' of training
    for epoch in range(1 + start_epoch, 1 + num_epochs):
        time_cnt = []

        # Generate new training data for each epoch
        _t = time.time()
        data = train_dataset.sample(sample_size=cfg['train_dataset_size'], graph_size=cfg['graph_size'])
        data = train_dataset
        t_sample += time.time() - _t
        # precompute baseline if necessary
        print("precompute baseline if necessary? ", type(baseline), )
        _t = time.time()
        data = baseline.wrap_dataset(data)
        t_baseline += time.time() - _t
        time_cnt.append(("bsl", time.time() - _t))
        # wrap into data iterator
        train_dl = DataLoader(data,
                              batch_size=cfg['train_batch_size'],
                              num_workers=cfg['num_workers'],
                              collate_fn=lambda x: x,  # identity -> returning simple list of instances
                              shuffle=True  # random shuffle instances
                              )

        # Put model in train mode
        policy.train()
        policy.set_decode_type("sampling")
        t_ep_start = time.time()

        # train over batches
        for batch_id, batch in enumerate(tqdm(train_dl, disable=no_p_bar)):

            result = train_batch(
                batch=batch,
                env=train_env,
                policy=policy,
                optimizer=optimizer,
                baseline=baseline,
                entropy_coeff=cfg['entropy_coeff'],
                bl_coeff=cfg['bl_coeff'],
                max_grad_norm=cfg['max_grad_norm']
            )
            n_episodes += cfg['train_batch_size']

            result['lr'] = optimizer.param_groups[0]['lr']
            monitor.log_train_data(result, n_episodes)

        ep_duration = time.time() - t_ep_start
        time_cnt.append(("train", ep_duration))
        t_epoch += ep_duration
        # if verbose > 1:
        logger.info(f"Finished epoch {epoch}, ({time.strftime('%H:%M:%S', time.gmtime(ep_duration))} s)")


        _t = time.time()
        # run validation
        val_result = validate(
            val_dataset, val_env, policy,
            batch_size=cfg['val_batch_size'],
            num_workers=cfg['num_workers'],
            disable_progress_bar=no_p_bar,
            render=render_val,
        )
        monitor.log_eval_data(val_result, n_episodes, mode="val")
        time_cnt.append(("valid", time.time() - t_ep_start))

        logger.info(f"Time benchmark {epoch}: {str(time_cnt)}")
        # train_eval_result = validate(
        #     train_dataset, val_env, policy,
        #     batch_size=cfg['val_batch_size'],
        #     num_workers=cfg['num_workers'],
        #     disable_progress_bar=no_p_bar,
        #     render=render_val,
        # )
        # monitor.log_eval_data(train_eval_result, n_episodes, mode="train_eval")

        cost, cost_std = val_result['cost'], val_result['cost_std']

        late_rate = val_result['late_rate']
        late_cost = val_result["late_cost"]
        late_num = val_result["late_num"]
        late_time = val_result["late_time"]
        if best_epoch < 0 or cost < best_cost:
            best_epoch, best_cost, best_cost_std = epoch, cost, cost_std

        # update baseline and scheduler
        baseline.epoch_callback(policy, epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()

        # save checkpoint
        monitor.save_data(epoch, n_episodes, gradient_step, ckpt_cb)
        # if verbose > 0:
        logger.info(f"Epoch #{epoch}: "
                    f"val_cost: {cost:.6f} ± {cost_std:.6f}, "
                    f"best_cost: {best_cost:.6f} ± {best_cost_std:.6f} in #{best_epoch}, "
                    f"late_num: {late_num:.6f} late_rate: {late_rate:.6f} late_cost: {late_cost:.6f}, "
                    f"late_time: {late_time:.6f}, "
                    f"valid_rate: {val_result['valid_rate']}, "
                    f"eval_gap: {val_result['avg_path_gap']}, "
                    f"eval_valid_gap: {val_result['avg_valid_path_gap']}, "
                    # f"train_gap: {train_eval_result['avg_path_gap']}, "
                    # f"train_valid_gap: {train_eval_result['avg_valid_path_gap']}, "
                    f"tm_gap: {val_result['avg_tm_gap']}")

    t_total = time.time() - t_start
    # checkpoint final model
    monitor.save_data(epoch, n_episodes, gradient_step, ckpt_cb, is_last=True)

    # gather training infos
    return {
        "time_total": t_total,
        "time/epoch": t_epoch/num_epochs,
        "time/bl_eval": t_baseline/num_epochs,
        "time/sample_ds": t_sample/num_epochs,
        "num_epochs": num_epochs,
        "total_num_episodes": n_episodes,
        "best_cost": best_cost,
        "best_cost_std": best_cost_std,
    }
