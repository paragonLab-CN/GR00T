# Closed-Loop Evaluation on Merlin

This guide reuses the existing LIBERO/LIBERO-Plus evaluator for the trusted-prefix
group-distillation branch. It does not introduce online LeWM selection, a new
simulator, or a distributed trajectory collector.

## Existing Entry Points

| Entry point | Purpose |
| --- | --- |
| `gr00t/eval/run_gr00t_server.py` | Load a checkpoint on CUDA and serve actions over ZMQ. |
| `gr00t/eval/rollout_policy.py` | Execute a policy in a Gymnasium simulator and report per-episode success. |
| `gr00t/eval/sim/LIBERO/libero_env.py` | Wrap LIBERO or its LIBERO-Plus fork using MuJoCo offscreen rendering. |
| `gr00t/eval/sim/LIBERO/setup_libero_plus.sh` | Build the dedicated LIBERO-Plus client environment. |
| `scripts/flare/run_eval_libero_plus_single_model.sh` | Start one policy server and evaluate one checkpoint across four suites. |
| `scripts/flare/eval_libero_plus_client.sh` | Generate/reuse a task manifest, run tasks, and aggregate JSON/CSV results. |
| `scripts/flare/run_eval_libero_plus_epoch50_fixed_44_20_fixed_noise.sh` | Reproduce the historical checkpoint-specific 112-task paired-noise protocol. |
| `scripts/flare/run_eval_libero_plus_epoch50_fixed_44_20_random_diffusion_noise.sh` | Historical checkpoint-specific evaluation with unfixed policy noise. |

Vanilla LIBERO also has `eval_libero10_client.sh`, `eval_libero90_client.sh`, and
`eval_spatial_client.sh` under `scripts/flare/`. Open-loop imitation-error
evaluation is available in `gr00t/eval/open_loop_eval.py`; it is not a substitute
for closed-loop task success.

For a new distillation checkpoint, use the generic single-model launcher rather
than a historical wrapper that asserts a specific step and GT/teacher batch mix.
The generic launcher already uses `candidate-k=1`: evaluate the trained student,
not an online best-of-K search algorithm.

## Runtime And Assets

Use one Linux NVIDIA GPU job/pod for the first evaluation. The server and simulator
client can run on the same node and communicate through `127.0.0.1:5555`, avoiding
cross-pod routing and port exposure. LIBERO uses MuJoCo/robosuite, not Isaac Sim.

- The repository documents 16 GB+ VRAM for inference. Allocate additional headroom
  for rendering and the actual checkpoint; this is not a measured memory bound.
- The default dGPU stack targets Python 3.10, CUDA 12.8, PyTorch 2.7.1, and
  flash-attn. Build an appropriate image or use a verified existing environment.
- Keep the root `.venv` for the model server and the separate
  `gr00t/eval/sim/LIBERO/libero_plus_uv/.venv` for the client. The client setup pins
  Python 3.10, PyTorch 2.5.1, Gymnasium 0.29.1, and MuJoCo 3.8.1; do not replace it
  with the root environment, whose Gymnasium version differs.
- The current `pyproject.toml` references a local x86_64 flash-attn wheel under
  `local_wheels/`. A fresh Git checkout alone does not provide that wheel. Provision
  a compatible wheel/environment before attempting `uv sync`. The model code can
  fall back to SDPA without flash-attn; the missing wheel blocks the default
  dependency installation, not every possible inference environment.
- Provision `external_dependencies/LIBERO-plus` at the pinned submodule revision,
  including its BDDL/task definitions, init files, task classification, and
  external assets. `assets.zip` comes from the authorized LIBERO-Plus source.
  The repository has a gitlink but no `.gitmodules` URL mapping; a fresh checkout
  needs an explicit clone of `https://github.com/sylvestf/LIBERO-plus` and checkout
  of `4976dc30028e805ff8094b55501d532c48fec182`, not just `git submodule update`.
- Provision the student checkpoint with its model, processor, statistics, and
  embodiment metadata. Also provision the authorized Cosmos-Reason2-2B backbone
  at `checkpoints/Cosmos-Reason2-2B`: the existing single-model script sets this
  local path explicitly.
- Provide NVIDIA EGL/OpenGL driver libraries inside the container, not just CUDA
  compute libraries. Set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` before imports.
  Verify offscreen rendering on the allocated GPU; a successful CUDA import alone
  does not prove EGL works.
- Recreate `libero_plus_config/config.yaml` for the actual container mount paths.
  Do not reuse the committed machine-specific absolute paths blindly.

Closed-loop student evaluation does not need LeWM, candidate caches, or training
demonstrations. Those are inputs to training, not to this evaluation entry point.
Keep checkpoints/results on approved persistent mounts or publish results to
approved storage; pod-local outputs are ephemeral.

This branch is training code, not a trained student checkpoint. A first wiring
smoke may use an existing LIBERO baseline and must be labeled as such. For the
default first spatial task, point `MODEL_PATH` at the `libero_spatial` subdirectory
of `nvidia/GR00T-N1.7-LIBERO`, not at the repository root or a checkpoint trained
only on `libero_10`. Method comparisons require actual trained weights. The
published baseline includes retained backbone tensors, but its processor is not
self-contained and the current constructor still loads the separate Cosmos
checkpoint before restoring the VLA state; a policy download alone is insufficient.

The LIBERO-Plus setup script rebuilds its client environment and config directory.
Only invoke it when that rebuild is intended. Export the repository path first:
its embedded config writer reads `LIBERO_PLUS_REPO` from the environment.

```bash
export GR00T_ROOT="$PWD"
export LIBERO_PLUS_REPO="$GR00T_ROOT/external_dependencies/LIBERO-plus"
bash gr00t/eval/sim/LIBERO/setup_libero_plus.sh
```

## Smoke Test Inside The GPU Job

Run from the checked-out GR00T root after provisioning the two environments and
assets. Replace the checkpoint placeholder with the actual persistent mount path.
These commands execute evaluation; they are not Merlin job-submission commands.

```bash
export GR00T_ROOT="$PWD"
export MODEL_PATH=/mnt/checkpoints/trusted_prefix_round1/checkpoint-10000
export SERVER_PY="$GR00T_ROOT/.venv/bin/python"
export GR00T_BACKBONE_MODEL_NAME=/mnt/checkpoints/Cosmos-Reason2-2B
export DEVICE=cuda:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$GR00T_ROOT/gr00t/eval/sim/LIBERO/libero_plus_config"
export CLIENT_PY="$GR00T_ROOT/gr00t/eval/sim/LIBERO/libero_plus_uv/.venv/bin/python"

nvidia-smi
"$GR00T_ROOT/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
"$CLIENT_PY" -c 'from libero.libero import benchmark; from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs; import gymnasium as gym; task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0); register_libero_envs(task_filter=task.name); env = gym.make("libero_sim/" + task.name); obs, _ = env.reset(seed=2026); print(task.name, obs["video.image"].shape); env.close()'

export N_EPISODES=1
export N_ENVS=1
export MAX_EPISODE_STEPS=720
export N_ACTION_STEPS=8
export SEED=2026
export DIFFUSION_SEED=2026
export TASK_LIMIT=1
export FULL_EVAL=0
export KEEP_VIDEOS=1
export RESUME=0
unset TASK_FILE_SOURCE
export EXPECTED_TASK_COUNT=1

bash scripts/flare/run_eval_libero_plus_single_model.sh \
  "$MODEL_PATH" trusted_prefix_round1_smoke 5555
```

The existing environment smoke script is useful for investigation, but currently
contains a hard-coded checkout path. The direct reset probe above derives the task
from the installed benchmark and avoids that path assumption.

The launchers accept prepared server/client Python paths and an explicit local
backbone path. They do not install missing assets or dependencies. A ready ZMQ
port is not a completed episode: require one actual episode in `summary_all.json`
and inspect the per-task `results:` log. With `candidate-k=1` and `N_ENVS=1`,
`episode_infos["episode_lengths"]` counts action-chunk requests, not individual
simulation steps; a value greater than one demonstrates feedback-driven calls.
Retain the smoke video for visual inspection before deleting temporary outputs.

For a standalone runtime diagnostic before provisioning policy assets, run:

```bash
python scripts/flare/preflight_libero_plus_runtime.py --check-network
```

This bounded probe checks real CUDA arithmetic, optional MuJoCo offscreen
rendering, and HTTPS access without downloading model/benchmark files. Even a
passing runtime precheck reports `closed_loop_completed=false` and
`ready_to_rollout=false`; it never substitutes for a GR00T/LIBERO episode. Do not
disable TLS verification to work around model-download certificate errors.

`KEEP_VIDEOS=0` deletes videos after each completed task; it does not disable video
encoding while the task is running. Include the video runtime dependencies and
temporary disk headroom even when retaining only summaries.

## Paired Full Evaluation

Use the same frozen task manifest for the baseline and every trained round. Mount
the historical manifest if exact historical comparability is required; a new
stratified sample is not automatically the identical protocol.

```bash
export TASK_FILE_SOURCE=/mnt/eval-protocols/libero_plus_112/tasks.tsv
export EXPECTED_TASK_COUNT=112
export N_EPISODES=10
export N_ENVS=1
export MAX_EPISODE_STEPS=720
export N_ACTION_STEPS=8
export SEED=2026
export DIFFUSION_SEED=2026
export TASK_LIMIT=0
export FULL_EVAL=0
export RESUME=1
export KEEP_VIDEOS=0

sha256sum "$TASK_FILE_SOURCE"
bash scripts/flare/run_eval_libero_plus_single_model.sh \
  "$MODEL_PATH" trusted_prefix_round1_seed2026 5555
```

The historical 112-task manifest SHA-256 is
`6412991d21db24aaacd402cdd35e716962df331112cdfa1a6201c20f306c3b3f`.
The client checks the requested task count, but the generic launcher does not
assert that historical hash; verify it separately when using that protocol.
There are 1,120 episodes per model in this protocol. `FULL_EVAL=1` instead selects
all classified benchmark tasks, not just those 112 tasks.

Results are written under `outputs/eval_libero_plus/<model_tag>/`:
`summary_all.json`, `task_success_rates_all.csv`, `tasks.tsv`, and per-task logs.
The JSON includes episode successes and suite/category/difficulty aggregates.
Record Git revision, checkpoint hash, task-manifest hash, benchmark revision,
environment versions, seeds, and action execution horizon alongside each run.
Use a new model tag for a changed checkpoint/protocol: resume checks are not a
complete checkpoint or environment identity check.

The fixed-noise protocol requires `N_ENVS=1` and assigns a server-side noise seed
per task, episode, and action-chunk request. Seeds do not guarantee bitwise equality
across hardware/software stacks. This wrapper's main reset path uses seeded
`OffScreenRenderEnv.reset()`, not the official precomputed init-state loading
shown in its standalone example; describe results as this repository's evaluation
protocol unless that distinction is resolved.

## Merlin Control-Plane Preflight

The available local control-plane CLI is `bytedcli merlin`, not a standalone
`merlin` executable. Use the current configured identity; do not put tokens,
cookies, SSH login details, or credentials in this repository or job logs.

```bash
bytedcli --site cn auth status
bytedcli --site cn --vregion seed merlin job list --page-size 1
bytedcli --site cn --vregion seed merlin quota group list --approved-only
bytedcli --site cn --vregion seed merlin quota group resources --group-id "$GROUP_ID"
```

Choose `GROUP_ID` from the nested `group.id`, not the membership record's `id`.
Confirm GPU quota, queue, image, and mount access before submission. Inspect a
known-working GPU job definition with the native `merlin job extract` command to
confirm the schema/image, but do not inherit its identities, credentials,
environment, mounts, worker counts, retries, or keep settings blindly. Verify all
duplicated resource sections and use a bounded, single-GPU runtime diagnostic
before provisioning the actual episode. Do not invent an image or resource schema
from the presence of a CLI command.

Scale later using disjoint frozen task manifests and distinct output namespaces.
Fixed-noise seeds depend on task indices in the manifest, so naive task splitting
changes those seeds. Preserve the original global task indices/seeds if claiming
the same paired protocol. No Ray conversion pipeline is needed for a first
single-node evaluation.

## Verification Status

Source inspection and `bash -n` succeeded for the generic launcher/client, the two
epoch-50 noise wrappers, and LIBERO-Plus setup. Local Merlin authentication status,
read-only job listing, and approved quota-group listing succeeded. The internal
Merlin skill reference was readable through the existing authorized browser
session; no authentication material was copied into this guide.

The launcher/client and standalone runtime probe have 62 passing Python 3.10 CPU
regression tests; the launchers also pass `bash -n`, and the new Python files pass
Ruff lint/format checks. These tests use fake policy/simulator and mocked GPU/network
success paths, not a real episode. The repository does not ship the configured
`.pre-commit-config.yaml`, so the documented pre-commit command cannot run.

At this source-validation checkpoint, no Merlin job was submitted or started. No
GPU inference, EGL rendering, benchmark rollout, checkpoint loading, capacity
reservation, or closed-loop success rate was verified. The local checkout and
authorized Merlin workspace did not contain the required evaluation environments
or benchmark/model assets. Runtime execution evidence must be recorded separately.
