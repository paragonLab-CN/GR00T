# result-to-claim C1/C4 epoch-9 re-adjudication (threadId 01a0f057-c650-7a40-b9f9-5d712178b502)
# model: gpt-5.5 xhigh. date 2026-09-30. Scorer-convention correction: headline = DEPLOYED epoch-9.

C1 -> YES, narrowed (medium-high). epoch-9 does NOT support "selects mostly sub-GT teachers"
  (GT beats all-4 in 44.4% = minority; GT vs best-cand MSE ~1.03x = tie). Holds as: filter selects
  self-consistent / near-demo teachers with little NEW info beyond GT. teacher_agreement=0.9435 is
  now the stronger pillar. Phrase: "selects self-consistent, near-demonstration teachers rather than
  demonstrably better-than-GT alternatives." AVOID "selects sub-GT teachers" (that's epoch-50 only).

C4 -> PARTIAL (medium / medium-low). epoch-9 does NOT support "broadly discards beneficial diversity"
  (keeps 64.5% @P50, 43.3% @P75). Holds NARROWLY: rejects the MOST-divergent valid paths (0/9173 @P90),
  over-admits redundant imitation (35.4%), admits jitter (6.8%). Phrase: "preserves some moderate
  beneficial diversity, but fails at the high-divergence tail and over-admits redundant near-GT teachers."

CENTRAL HOOK: original ("selects redundant + discards rare beneficial") TOO BROAD for epoch-9.
Honest thesis: "the deployed filter is less a catastrophic bad-teacher selector than a conservative
self-consistency selector: it mostly preserves near-GT redundant teachers, admits some jitter, and
loses the most-divergent valid alternatives where exploration value would be highest."
Paper REMAINS writable; center of gravity shifts from "selects worse teachers" to "adds little useful
signal beyond GT and is poorly aligned with harvesting rare valid high-divergence alternatives."
Epoch-50 = sensitivity only, not headline.
