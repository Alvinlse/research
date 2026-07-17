# Referee manual — SELF-AUTHORED (Exp 51 Phase A; do not hand-edit, regenerate via pins/manual_author.py)

<!-- PROMPT-START: everything below this line is sent to the referee -->
PRECEDENTS from past allocations (apply when the WHEN clause matches; cite the precedent id in your justification):

P1. WHEN free_pool_gpus >=8 and n_jobs >=16 and upcoming_prod_jobs ==2 and (current_allocated_gpus + reserved_gpus) < total_available_gpus - 3:
    Reserve at least 3 GPUs for incoming prod jobs. For each additional upcoming prod job beyond 2, increase the reserve by 1 GPU up to a maximum of 5 reserved GPUs.
    [learned: Reducing the minimum reserve from 4 to 3 improved besteffort job outcomes without significantly affecting production SLA.]
