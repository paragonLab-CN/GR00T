# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
import random
import sys
import time
from typing import Any
import uuid

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
from gr00t.utils.determinism import seed_everything
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import tyro


class TrtMode(str, Enum):
    """TensorRT inference modes."""

    N17_FULL_PIPELINE = "n17_full_pipeline"
    VIT_LLM_ONLY = "vit_llm_only"
    ACTION_HEAD = "action_head"


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
        input_pix_fmt: Input pixel format
        crf: Constant Rate Factor for video compression (lower = better quality)
        thread_type: Threading strategy for video encoding
        thread_count: Number of threads to use for encoding
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    n_action_steps: int = 8


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def get_simpler_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

        register_simpler_envs()
        return gym.make(env_name)

    return env_fn


def get_libero_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

        # env_name is "libero_sim/<task_name>"; only register the target task.
        # LIBERO-plus registers thousands of task variants, and registering
        # all of them per rollout process makes startup needlessly slow.
        task_name = env_name.split("/", 1)[1] if "/" in env_name else env_name
        register_libero_envs(task_filter=task_name)
        return gym.make(env_name)

    return env_fn


def get_gym_env(env_name: str, env_idx: int, total_n_envs: int):
    """Create Ray environment factory function without wrappers."""

    env_embodiment = get_embodiment_tag_from_env_name(env_name)

    if env_embodiment in (EmbodimentTag.SIMPLER_ENV_GOOGLE, EmbodimentTag.SIMPLER_ENV_WIDOWX):
        env_fn = get_simpler_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.LIBERO_PANDA,):
        env_fn = get_libero_env_fn(env_name)

    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()


def create_eval_env(
    env_name: str, env_idx: int, total_n_envs: int, wrapper_configs: WrapperConfigs
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        wrapper_configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """

    env = get_gym_env(env_name, env_idx, total_n_envs)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


class _RobustAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    """AsyncVectorEnv that tolerates variable-shaped info arrays across envs.

    Gymnasium's default _add_info pre-allocates a numpy array based on the
    first env's value shape and then assigns subsequent envs into it.  When
    envs return differently-shaped values (e.g. variable-length contact arrays)
    the assignment raises ValueError.  We catch that and fall back to a plain
    Python list for that key so the rest of the step can proceed normally.
    """

    def _add_info(self, infos, info, env_num):
        for k, v in info.items():
            if k not in infos:
                infos[k] = [None] * self.num_envs
                infos[f"_{k}"] = np.zeros(self.num_envs, dtype=bool)
            if isinstance(infos[k], np.ndarray):
                try:
                    infos[k][env_num] = v
                except (ValueError, TypeError):
                    lst = list(infos[k])
                    lst[env_num] = v
                    infos[k] = lst
            else:
                infos[k][env_num] = v
            infos[f"_{k}"][env_num] = True
        return infos
    
def _seed_candidate_sampling(seed: int) -> None:
    """Seed local RNGs before each GR00T candidate sampling call."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_saveable_cpu(x):
    """Recursively convert observations/actions to CPU-saveable objects."""
    if torch.is_tensor(x):
        return x.detach().cpu().clone()

    if isinstance(x, np.ndarray):
        # object array 里面可能嵌套 dict/list，直接 torch.from_numpy 会失败
        if x.dtype == object:
            return [_to_saveable_cpu(v) for v in x.tolist()]
        return torch.from_numpy(np.array(x, copy=True))

    if isinstance(x, dict):
        return {k: _to_saveable_cpu(v) for k, v in x.items()}

    if isinstance(x, list):
        return [_to_saveable_cpu(v) for v in x]

    if isinstance(x, tuple):
        return tuple(_to_saveable_cpu(v) for v in x)

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    # 兜底：不认识的类型先转成字符串，避免 torch.save 崩掉
    return repr(x)


def sample_k_candidate_actions(
    policy: BasePolicy,
    observations: Any,
    k: int = 4,
    base_seed: int = 2026,
    rollout_step: int = 0,
    verbose: bool = False,
    candidate_save_dir: str | None = None,
):
    """Sample K candidate action trajectories for the same observation.

    Current MVP behavior:
    1. Repeatedly call policy.get_action(observations) K times.
    2. Print/debug the stacked candidate shape [B, K, T, D].
    3. Return candidate 0 to env.step for now.

    Later, replace `chosen_idx = 0` with the LeWM best-index.
    """
    if k <= 1:
        return policy.get_action(observations)

    candidate_actions: list[dict[str, np.ndarray]] = []
    candidate_infos: list[Any] = []

    for candidate_idx in range(k):
        # Make each candidate use a different initial diffusion noise.
        # rollout_step is included so the same candidate index does not reuse
        # exactly the same RNG state at every environment step.
        seed = int(base_seed) + int(rollout_step) * 100000 + int(candidate_idx)
        _seed_candidate_sampling(seed)

        actions_i, info_i = policy.get_action(observations)
        candidate_actions.append(actions_i)
        candidate_infos.append(info_i)
    
    # 每个 action 字段保存成 [B, K, T, D_key]
    stacked_candidate_actions = {
        action_key: np.stack(
            [np.asarray(actions_i[action_key]).copy() for actions_i in candidate_actions],
            axis=1,
        )
        for action_key in candidate_actions[0].keys()
    }

    if verbose:
        print(f"\n[FLARE debug] rollout_step={rollout_step}, sampled K={k} candidates")
        for action_key, stacked in stacked_candidate_actions.items():
            print(f"[FLARE debug] {action_key}: candidate_actions shape = {stacked.shape}")
            if k >= 2:
                max_abs_diff = np.max(np.abs(stacked[:, 0] - stacked[:, 1]))
                print(
                    f"[FLARE debug] max |candidate 0 - candidate 1| "
                    f"for {action_key} = {max_abs_diff:.6f}"
                )

    chosen_idx = 0

    if candidate_save_dir:
        save_dir = Path(candidate_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"step_{int(rollout_step):06d}.pt"

        stacked_candidate_actions_torch = {
            action_key: torch.from_numpy(stacked.copy())
            for action_key, stacked in stacked_candidate_actions.items()
        }

        torch.save(
            {
                "rollout_step": int(rollout_step),
                "candidate_k": int(k),
                "chosen_idx": int(chosen_idx),
                "action_keys": list(stacked_candidate_actions_torch.keys()),

                # 这个 observation 就是生成 K 条候选动作时的当前观测
                "observations": _to_saveable_cpu(observations),

                # 每个 action 字段形状是 [B, K, T, 1]
                "candidate_actions": stacked_candidate_actions_torch,
            },
            save_path,
        )

        if verbose:
            print(f"[FLARE debug] saved candidate actions to: {save_path}")

    return candidate_actions[chosen_idx], candidate_infos[chosen_idx]


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
    seed: int | None = None,
    candidate_k: int = 1,
    candidate_base_seed: int = 2026,
    candidate_verbose: bool = False,
    candidate_save_dir: str | None = None,
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy: Policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        seed: If set, forwards per-env seeds (``seed+i``) to the first
            ``env.reset`` so each sub-env is reproducible. Should be paired
            with :func:`gr00t.utils.determinism.seed_everything` upstream to
            also constrain policy-side RNGs.
    Returns:
        Collection results from running the episodes
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = _RobustAsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Storage for results
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    current_rewards = [0.0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)

    # Initial reset; if a seed is provided, give each sub-env a distinct but
    # deterministic seed so that parallel workers don't all start from the
    # same initial state while still being run-to-run reproducible.
    if seed is not None:
        reset_seeds = [int(seed) + i for i in range(n_envs)]
        observations, _ = env.reset(seed=reset_seeds)
    else:
        observations, _ = env.reset()
    policy.reset()
    i = 0

    pbar = tqdm(total=n_episodes, desc="Episodes")
    while completed_episodes < n_episodes:
        actions, _ = sample_k_candidate_actions(
            policy=policy,
            observations=observations,
            k=candidate_k,
            base_seed=candidate_base_seed,
            rollout_step=i,
            verbose=candidate_verbose and i < 3,
            candidate_save_dir=candidate_save_dir,
        )
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
        # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
        i += 1
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                if "valid" in env_infos:
                    episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                # Accumulate per-episode results. Both lists are captured
                # BEFORE the per-env trackers are reset to 0 below — without
                # this ordering downstream consumers silently see
                # episode_length=0 / episode_reward=0.0.
                episode_lengths.append(current_lengths[env_idx])
                episode_rewards.append(float(current_rewards[env_idx]))
                episode_successes.append(current_successes[env_idx])
                # Reset trackers for this environment.
                current_successes[env_idx] = False
                # only update completed_episodes if valid
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    # envs don't return valid
                    completed_episodes += 1
                    pbar.update(1)
                # Reset with `0.0` to match the `[0.0] * n_envs` init and the
                # `float(...)` cast on line 347; otherwise the per-env entry's
                # static type silently flips int <-> float across iterations.
                current_rewards[env_idx] = 0.0
                current_lengths[env_idx] = 0
        observations = next_obs
    pbar.close()

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    # `current_lengths[env_idx] += 1` runs before each termination check
    # above, so any captured episode must have stepped at least once.
    assert all(length >= 1 for length in episode_lengths), (
        f"Internal invariant violated: rollout produced zero-length episode(s) "
        f"in {episode_lengths!r}."
    )

    # Surface the per-episode length and reward that were tracked locally so
    # downstream metrics (SimplerEnv / LIBERO / Robocasa / Wholebody) can
    # read them off episode_infos instead of silently falling back to 0.
    # Planted BEFORE the "valid" filter so they get filtered in lockstep
    # with the other episode_infos fields.
    episode_infos["episode_lengths"] = episode_lengths
    episode_infos["episode_rewards"] = episode_rewards

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    return env_name, episode_successes, episode_infos


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    trt_engine_path: str = "",
    trt_mode: TrtMode = TrtMode.N17_FULL_PIPELINE,
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        gr00t_policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=0,
        )
        if trt_engine_path:
            deploy_dir = str(Path(__file__).resolve().parents[2] / "scripts" / "deployment")
            if deploy_dir not in sys.path:
                sys.path.insert(0, deploy_dir)
            from trt_model_forward import setup_tensorrt_engines

            setup_tensorrt_engines(gr00t_policy, trt_engine_path, mode=trt_mode)
        policy = Gr00tSimPolicyWrapper(gr00t_policy)
    return policy


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    video_dir: str | None = None,
    trt_engine_path: str = "",
    trt_mode: TrtMode = TrtMode.N17_FULL_PIPELINE,
    seed: int | None = None,
    candidate_k: int = 1,
    candidate_base_seed: int = 2026,
    candidate_verbose: bool = False,
    candidate_save_dir: str | None = None,
):
    # seed_everything resolves `None` via the GR00T_EVAL_SEED env var and is a
    # no-op when that is also unset, so the historical non-deterministic
    # behavior is preserved by default. Returns the effective seed (or None)
    # which we forward to env.reset below.
    seed = seed_everything(seed)

    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    if video_dir is None:
        if model_path:
            parts = model_path.split("/")
            model_slug = parts[-3] if len(parts) >= 3 else parts[-1]
            video_dir = f"/tmp/sim_eval_videos_{model_slug}_ac{n_action_steps}_{uuid.uuid4()}"
        else:
            video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = create_gr00t_sim_policy(
        model_path,
        embodiment_tag,
        policy_client_host,
        policy_client_port,
        trt_engine_path=trt_engine_path,
        trt_mode=trt_mode,
    )

    results = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
        seed=seed,
        candidate_k=candidate_k,
        candidate_base_seed=candidate_base_seed,
        candidate_verbose=candidate_verbose,
        candidate_save_dir=candidate_save_dir,
    )
    print("Video saved to: ", wrapper_configs.video.video_dir)
    return results


@dataclass
class RolloutConfig:
    """Configuration for rollout policy evaluation."""

    max_episode_steps: int = 504
    """Maximum number of steps per episode."""

    n_episodes: int = 50
    """Number of episodes to run."""

    model_path: str = ""
    """Path to model checkpoint."""

    policy_client_host: str = ""
    """Host for policy client."""

    policy_client_port: int | None = None
    """Port for policy client."""

    env_name: str = "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    """Environment name."""

    n_envs: int = 8
    """Number of parallel environments."""

    n_action_steps: int = 8
    """Number of action steps."""

    video_dir: str | None = None
    """Directory to save videos. If None, uses /tmp/sim_eval_videos_<env>_<uuid>."""

    trt_engine_path: str = ""
    """Path to TRT engine directory. If set, uses TRT inference instead of PyTorch."""

    trt_mode: TrtMode = TrtMode.N17_FULL_PIPELINE
    """TRT mode: 'n17_full_pipeline' (all engines), 'vit_llm_only', or 'action_head'."""

    seed: int | None = None
    """Optional seed for deterministic evaluation. When set, seeds Python /
    NumPy / torch / cuda RNGs, enables cuDNN determinism, and forwards
    per-env seeds to the sim envs. If left as ``None``, falls back to the
    ``GR00T_EVAL_SEED`` env var; if that is also unset, keeps the historical
    non-deterministic behavior."""

    candidate_k: int = 1
    """Number of candidate action trajectories sampled for the same observation.
    Use 1 for the original GR00T behavior; use 4/8 for FLARE-VLA candidate sampling."""

    candidate_base_seed: int = 2026
    """Base seed used to create different initial denoising noises for candidates."""

    candidate_verbose: bool = False
    """If True, print candidate action shapes and candidate differences for first rollout steps."""

    candidate_save_dir: str | None = None
    """Directory used to save FLARE candidate action .pt files. If None, nothing is saved."""

if __name__ == "__main__":
    args = tyro.cli(RolloutConfig)

    # validate policy configuration
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
        "If all 3 arguments are provided, explicitly choose one:\n"
        '  - To use policy client: set --policy-client-host and --policy-client-port, and set --model-path ""\n'
        '  - To use model path: set --model-path, and set --policy-client-host "" (and leave --policy-client-port unset)'
    )

    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        video_dir=args.video_dir,
        trt_engine_path=args.trt_engine_path,
        trt_mode=args.trt_mode,
        seed=args.seed,
        candidate_k=args.candidate_k,
        candidate_base_seed=args.candidate_base_seed,
        candidate_verbose=args.candidate_verbose,
        candidate_save_dir=args.candidate_save_dir,
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))
