# Trusted-Prefix Group Self-Distillation

This opt-in training path reuses the existing offline K-candidate generator,
LeWM scoring, teacher-quality audit, and flow-matching student. It is grouped
self-distillation inspired by GRPO, not PPO likelihood-ratio GRPO. No model
architecture, critic, simulator, or deployment-time WM call is added.

## What Changes

For each demonstrated observation, the existing cache contains K short action
chunks, their per-step predicted-future errors, and a demonstrated-action oracle
error. The oracle measures how well LeWM predicts the demonstration's future
when given demonstrated actions. It is a reliability proxy, not a bound on
counterfactual error, physical safety, or task success.

The group builder keeps only a continuous candidate prefix for which both the
oracle error and candidate-minus-oracle error meet calibrated tolerances. The
first untrusted step terminates supervision; later steps do not re-enter it.
Action bounds and finite-value checks remain active. At least two candidates
must survive, with sufficient score separation; otherwise the group abstains.

By default, group probabilities compare all eligible candidates over their
shared shortest trusted prefix. An untrusted tail therefore cannot decide
their ranking, and unequal prefix lengths do not bias score comparisons. Each
sampled candidate retains its own supervision length. Candidates are never
averaged or spliced into a synthetic action sequence.

The loader draws one candidate per shard materialization according to q. This
is a Monte Carlo estimate of the weighted group loss without a K-fold training
batch expansion. Group abstention uses the original full GT target, not the old
top1 teacher. Per-sample valid-coordinate normalization makes GT/teacher loss
weights independent of supervision length:

```text
L = (1 - lambda) mean_GT(L_FM)
    + lambda mean_teacher(E[k ~ q] L_FM(candidate_k, trusted_prefix_k))
```

`--group-teacher-loss-weight` sets lambda (default 0.3). When an entire global
batch has only one effective source, that source receives full weight. The
collator enables group metadata for every rank, including all-GT batches;
normalization masses are shared across DDP ranks. Legacy sidecars retain their
original batch-global coordinate reduction and do not add these collectives.

## Build From Trusted Cached Scores

Use a **new output directory**. The original index, top1 targets, and summaries
are preserved for compatibility; an optional `group_teacher` object stores all
candidates, q, prefix lengths, selection settings, round, and policy/WM hashes.

```bash
python scripts/flare/build_group_teacher.py \
  --source-dir /data/top1_distillation \
  --score-parts-dir /data/scored/score_parts \
  --policy-provenance /data/policy_provenance.json \
  --output-dir /data/trusted_prefix_groups \
  --score-metric mse --score-scope shared-prefix \
  --oracle-max-mse "$CALIBRATED_ORACLE_MSE" \
  --max-paired-gap "$CALIBRATED_PAIRED_GAP" \
  --min-score-span "$CALIBRATED_MIN_SCORE_SPAN" \
  --score-std-floor "$CALIBRATED_SCORE_NOISE_FLOOR" \
  --min-prefix 2
```

Calibrate thresholds on an episode-disjoint training calibration split, freeze
them before evaluation, and reuse them across rounds. Numeric defaults are not
a calibrated uncertainty estimate. `--score-scope full`, `--fixed-prefix 8`,
and `--no-oracle-anchor` support controls against ordinary group weighting.

Legacy candidates did not record a policy hash. Their provenance manifest must
be supplied from reliable generation records, not guessed from a mutable path:

```json
{
  "format": "flare_group_policy_provenance_v1",
  "round_id": 0,
  "policy_checkpoint_sha256": "<64-character snapshot hash>",
  "candidates": [
    {"sample_id": "<sample>", "candidate_sha256": "<64-character file hash>"}
  ]
}
```

The manifest must cover every score part, not just selected teachers. Only a
single frozen policy snapshot is supported per round. Old four-suite caches
from four different policy directories cannot be relabeled as one policy.
PyTorch artifacts must be trusted; without the original candidate files, legacy
attestation cannot independently verify all candidate actions or scores.

Train with the existing finetune entry point, the new group directory, full GT
replay, and preferably fixed per-device source counts. Sampling fractions and
the explicit teacher loss coefficient are different controls. For example,
44 GT / 20 teacher samples do not imply lambda = 20/64.

## Refresh Rounds

The example plan uses a single frozen student to generate the full LIBERO-Plus
manifest. The generator and scorer have explicit policy-hash options for this
path; default legacy per-suite checkpoint checks remain unchanged. The scorer
still requires the same validated H8 frame bank and LeWM checkpoint.
The initial sampling policy must be a LIBERO-finetuned checkpoint with
`LIBERO_PANDA` processor/embodiment metadata, not the raw GR00T-N1.7-3B base
checkpoint. Configure one shared policy for all suites in that round.

```bash
python scripts/flare/run_group_distillation_rounds.py \
  --plan scripts/flare/group_distillation_round_plan.example.json --dry-run
```

The example contains asset and calibration placeholders and cannot execute
until these are configured. It runs the existing audit and top1 builder each
round before adding group targets, so refreshed targets really come from that
round's candidates. The old suite checkpoint root is used only by the existing
builder's action/processor shape checks; it is not the sampling policy.

Actual execution, after adapting the plan, uses the same command without
`--dry-run`. It refuses existing round directories, missing or mismatched policy
hashes in generated summaries/candidates/score parts, mutated behavior
checkpoints, missing student checkpoints, and refresh to unchanged weight
artifacts. Snapshot hashes include inference configuration but exclude training
logs; a separate weight-only hash prevents configuration changes from counting
as student updates. Status and both identities are recorded in each round's
`round_status.json`. A changed artifact hash is not evidence of improved policy
quality. It does not submit Merlin jobs, resume partial rounds, or
claim exact mid-epoch candidate replay after a training interruption.

Two or three rounds are a starting experiment, not a guaranteed improvement.
Keep the total candidate and student-update budgets matched against static
teachers and larger one-shot searches. Evaluate the student's closed-loop task
success, not only predicted-future scores. See
[Merlin evaluation](trusted_prefix_group_distillation_merlin_eval.md).

## Scope And Verification

This branch supervises at most the existing eight-step candidate prefix. It
does not extend a 40-step internal model horizon to hundreds of open-loop
actions or remove environment feedback. Deployment remains a single-policy
call: the method adds no online latency, but does not automatically lower the
existing offline-teacherfilter baseline's latency.

CPU tests cover selection, sidecar mapping/provenance, sampled targets, masks,
loss weighting, checkpoint loading, and round orchestration. CPU mocks do not
establish GPU throughput, real distributed training, WM ranking accuracy, or
Merlin task success. The GPU experiment and final paired rollout are required
before claiming empirical improvements.
