# Referee manual — SELF-AUTHORED (Exp 51 Phase A; do not hand-edit, regenerate via pins/manual_author.py)

<!-- PROMPT-START: everything below this line is sent to the referee -->
PRECEDENTS from past allocations (apply when the WHEN clause matches; cite the precedent id in your justification):

P1. WHEN free_gpus >= 1 AND incoming_prod_count > 0:
    reserve = 1
    [learned: Reserving 1 GPU when free GPUs are available and prod jobs are incoming improved utilization without breaking SLAs.]
