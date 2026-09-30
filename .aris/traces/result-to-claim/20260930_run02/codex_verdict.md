# result-to-claim C2/C3 re-adjudication (threadId 01a0f034-5600-7470-83ec-3db8ba4cc996)
# model: gpt-5.5, xhigh. date 2026-09-30. Data: teacher_scoring_v2 (epoch-9 scorer, N=487,880, distilled into gt_replay_44_20).

C2 -> yes (0.8). 138 features organize into few interpretable ~orthogonal axes; diversity decoupled
  from oracle-gap (rho=0.056) and action-mse (rho=0.013). Use "approximately orthogonal/decoupled",
  NOT "independent" (no ICA/MI).
C3 -> no / evidenced negative (0.85). No feature block linearly predicts per-task downstream success
  (all p>0.39, n=35; oracle_gap rho=-0.03; exact-match flips sign). Revised claim: per-example filter
  diagnostics do not transfer straightforwardly to per-task downstream generalization.

Framing: C3 null WEAKENS "misaligned because under-weights oracle gap". Safer thesis = selection
geometry is interpretable (C2) but downstream utility not captured by per-task aggregates (C3 null);
descriptive misalignment (C1/C4/C5) + D1 underperformance are the load-bearing findings.
