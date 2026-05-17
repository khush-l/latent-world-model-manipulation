"""Local launcher for SoftAgent algorithms on this repository's SoftGym copy.

The original SoftAgent scripts are vendored in this directory. This wrapper
keeps their algorithm implementations intact while making local simulation runs
less surprising: it defaults to one variation, headless execution, and explicit
cache controls instead of using this repository's 1000-variation SoftGym
defaults.
"""

from __future__ import print_function

import argparse
import copy
import os

from softgym.registered_env import env_arg_dict


SUPPORTED_ALGORITHMS = ("cem", "curl", "drq", "planet", "mvp")


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def add_common_args(parser):
    parser.add_argument("--algorithm", choices=SUPPORTED_ALGORITHMS, default="cem")
    parser.add_argument("--env-name", default="ClothFlatten")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--test-episodes", default=1, type=int)

    parser.add_argument("--observation-mode", default=None)
    parser.add_argument("--action-mode", default=None)
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--camera-name", default="default_camera")
    parser.add_argument("--render", default=True, type=str2bool)
    parser.add_argument("--headless", default=True, type=str2bool)
    parser.add_argument("--deterministic", default=False, type=str2bool)
    parser.add_argument("--num-variations", default=1, type=int)
    parser.add_argument("--horizon", default=None, type=int)
    parser.add_argument("--action-repeat", default=None, type=int)
    parser.add_argument("--num-picker", default=None, type=int)
    parser.add_argument("--use-cached-states", default=False, type=str2bool)
    parser.add_argument("--save-cached-states", default=False, type=str2bool)


def add_algorithm_args(parser):
    parser.add_argument("--max-iters", default=2, type=int)
    parser.add_argument("--timestep-per-decision", default=200, type=int)
    parser.add_argument("--use-mpc", default=True, type=str2bool)

    parser.add_argument("--num-train-steps", default=None, type=int)
    parser.add_argument("--num-seed-steps", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--replay-ratio", default=None, type=float)
    parser.add_argument("--replay-buffer-capacity", default=None, type=int)
    parser.add_argument("--eval-frequency", default=None, type=int)
    parser.add_argument("--log-interval", default=None, type=int)
    parser.add_argument("--im-size", default=128, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-video", default=True, type=str2bool)
    parser.add_argument("--save-model", default=False, type=str2bool)
    parser.add_argument("--save-tb", default=False, type=str2bool)

    parser.add_argument("--collect-interval", default=100, type=int)
    parser.add_argument("--test-interval", default=10, type=int)
    parser.add_argument("--train-epoch", default=None, type=int)
    parser.add_argument("--planning-horizon", default=24, type=int)
    parser.add_argument("--use-value-function", default=False, type=str2bool)

    parser.add_argument("--config-key", default="sac_pixels_cloth_corner_softgym")
    parser.add_argument("--sac-module", default="sac_v2")
    parser.add_argument("--sac-agent-module", default="sac_agent_v2")
    parser.add_argument("--mvp-batch-b", default=1, type=int)
    parser.add_argument("--mvp-eval-n-envs", default=1, type=int)
    parser.add_argument("--mvp-eval-max-steps", default=None, type=int)
    parser.add_argument("--mvp-eval-max-trajectories", default=None, type=int)


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_algorithm_args(parser)
    return parser.parse_args()


def default_log_dir(algorithm):
    return os.path.join("..", "data", "softagent", algorithm)


def default_exp_name(algorithm):
    names = {
        "cem": "CEM",
        "curl": "CURL_SAC",
        "drq": "DrQ_SAC",
        "planet": "PlaNet",
        "mvp": "MVP_QPG",
    }
    return names[algorithm]


def env_override_dict(args, default_observation_mode):
    overrides = {
        "env_kwargs_render": args.render,
        "env_kwargs_headless": args.headless,
        "env_kwargs_deterministic": args.deterministic,
        "env_kwargs_num_variations": args.num_variations,
        "env_kwargs_use_cached_states": args.use_cached_states,
        "env_kwargs_save_cached_states": args.save_cached_states,
        "env_kwargs_camera_name": args.camera_name,
        "env_kwargs_observation_mode": args.observation_mode or default_observation_mode,
    }
    if args.action_mode is not None:
        overrides["env_kwargs_action_mode"] = args.action_mode
    if args.render_mode is not None:
        overrides["env_kwargs_render_mode"] = args.render_mode
    if args.horizon is not None:
        overrides["env_kwargs_horizon"] = args.horizon
    if args.action_repeat is not None:
        overrides["env_kwargs_action_repeat"] = args.action_repeat
    if args.num_picker is not None:
        overrides["env_kwargs_num_picker"] = args.num_picker
    return overrides


def base_variant(args, algorithm, default_observation_mode):
    vv = {
        "exp_name": args.exp_name or default_exp_name(algorithm),
        "env_name": args.env_name,
        "log_dir": args.log_dir or default_log_dir(algorithm),
        "test_episodes": args.test_episodes,
        "seed": args.seed,
        "env_kwargs": env_arg_dict[args.env_name],
    }
    vv.update(env_override_dict(args, default_observation_mode))
    return vv


def run_cem(args):
    from experiments.run_cem import run_task

    vv = base_variant(args, "cem", default_observation_mode="key_point")
    vv.update({
        "max_iters": args.max_iters,
        "timestep_per_decision": args.timestep_per_decision,
        "use_mpc": args.use_mpc,
    })
    run_task(vv, vv["log_dir"], vv["exp_name"])


def run_curl(args):
    from experiments.run_curl import (
        clip_obs,
        get_actor_critic_lr,
        get_alpha_lr,
        get_lr_decay,
        reward_scales,
        run_task,
    )

    vv = base_variant(args, "curl", default_observation_mode="key_point")
    obs_mode = vv["env_kwargs_observation_mode"]
    vv.update({
        "algorithm": "CURL",
        "alpha_fixed": False,
        "init_temperature": 0.1,
        "replay_buffer_capacity": args.replay_buffer_capacity or 100000,
        "batch_size": args.batch_size or 128,
        "save_tb": args.save_tb,
        "save_video": args.save_video,
        "save_model": args.save_model,
        "actor_lr": get_actor_critic_lr(args.env_name, obs_mode),
        "critic_lr": get_actor_critic_lr(args.env_name, obs_mode),
        "alpha_lr": get_alpha_lr(args.env_name, obs_mode),
        "lr_decay": get_lr_decay(args.env_name, obs_mode),
        "scale_reward": reward_scales[args.env_name],
        "clip_obs": clip_obs[args.env_name] if obs_mode == "key_point" else None,
    })
    if args.num_train_steps is not None:
        vv["num_train_steps"] = args.num_train_steps
    if args.log_interval is not None:
        vv["log_interval"] = args.log_interval
    run_task(vv, vv["log_dir"], vv["exp_name"])


def run_drq(args):
    from experiments.run_drq import clip_obs, get_critic_lr, get_lr_decay, reward_scales
    from drq.train import run_task

    vv = base_variant(args, "drq", default_observation_mode="cam_rgb")
    obs_mode = vv["env_kwargs_observation_mode"]
    vv.update({
        "algorithm": "Drq",
        "alpha_fixed": False,
        "init_temperature": 0.1,
        "replay_buffer_capacity": args.replay_buffer_capacity or 100000,
        "batch_size": args.batch_size or 128,
        "num_train_steps": args.num_train_steps or 1000000,
        "im_size": args.im_size,
        "device": args.device,
        "save_video": args.save_video,
        "save_model": args.save_model,
        "log_save_tb": args.save_tb,
        "actor_lr": get_critic_lr(args.env_name, obs_mode),
        "critic_lr": get_critic_lr(args.env_name, obs_mode),
        "lr_decay": get_lr_decay(args.env_name, obs_mode),
        "scale_reward": reward_scales[args.env_name],
        "clip_obs": clip_obs[args.env_name] if obs_mode == "key_point" else None,
    })
    if args.num_seed_steps is not None:
        vv["num_seed_steps"] = args.num_seed_steps
    if args.eval_frequency is not None:
        vv["eval_frequency"] = args.eval_frequency
    if args.log_interval is not None:
        vv["log_interval"] = args.log_interval
        vv["log_frequency_step"] = args.log_interval
    run_task(vv, vv["log_dir"], vv["exp_name"])


def run_planet(args):
    from planet.train import run_task

    vv = base_variant(args, "planet", default_observation_mode="cam_rgb")
    vv.update({
        "algorithm": "planet",
        "collect_interval": args.collect_interval,
        "test_interval": args.test_interval,
        "batch_size": args.batch_size or 128,
        "train_epoch": args.train_epoch or 1200,
        "planning_horizon": args.planning_horizon,
        "use_value_function": args.use_value_function,
        "image_dim": args.im_size,
        "symbolic_env": vv["env_kwargs_observation_mode"] != "cam_rgb",
    })
    horizon = env_arg_dict[args.env_name]["horizon"]
    vv["test_episodes"] = max(1, 900 // horizon)
    vv["episodes_per_loop"] = max(1, 900 // horizon)
    run_task(vv, vv["log_dir"], vv["exp_name"])


def run_mvp(args):
    from rlpyt_cloth.rlpyt.experiments.scripts.dm_control.qpg.sac.train.softgym_sac import run_task
    from rlpyt.experiments.configs.dm_control.qpg.sac.softgym_sac import configs

    vv = base_variant(args, "mvp", default_observation_mode="cam_rgb")
    base_config = configs[args.config_key]
    runner = copy.deepcopy(base_config["runner"])
    sampler = copy.deepcopy(base_config["sampler"])
    algo = copy.deepcopy(base_config["algo"])

    horizon = args.horizon or 20
    train_steps = args.num_train_steps if args.num_train_steps is not None else 1
    eval_max_steps = args.mvp_eval_max_steps if args.mvp_eval_max_steps is not None else horizon

    runner.update({
        "n_steps": train_steps,
        "log_interval_steps": args.log_interval or max(1, train_steps),
    })
    sampler.update({
        "batch_T": 1,
        "batch_B": args.mvp_batch_b,
        "max_decorrelation_steps": 0,
        "eval_n_envs": args.mvp_eval_n_envs,
        "eval_max_steps": eval_max_steps,
        "eval_max_trajectories": args.mvp_eval_max_trajectories or args.test_episodes,
    })
    algo.update({
        "batch_size": args.batch_size or 1,
        "replay_ratio": args.replay_ratio if args.replay_ratio is not None else 1,
    })

    vv.update({
        "algorithm": "qpg",
        "config_key": args.config_key,
        "sac_module": args.sac_module,
        "sac_agent_module": args.sac_agent_module,
        "env_kwargs_num_picker": args.num_picker or 1,
        "env_kwargs_horizon": horizon,
        "env_kwargs_action_mode": args.action_mode or "picker_qpg",
        "runner": runner,
        "sampler": sampler,
        "algo": algo,
    })
    run_task(vv, vv["log_dir"], vv["exp_name"])


def main():
    args = parse_args()
    runners = {
        "cem": run_cem,
        "curl": run_curl,
        "drq": run_drq,
        "planet": run_planet,
        "mvp": run_mvp,
    }
    runners[args.algorithm](args)


if __name__ == "__main__":
    main()
