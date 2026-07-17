# Referee manual — SELF-AUTHORED (Exp 51 Phase A; do not hand-edit, regenerate via pins/manual_author.py)

<!-- PROMPT-START: everything below this line is sent to the referee -->
PRECEDENTS from past allocations (apply when the WHEN clause matches; cite the precedent id in your justification):

P1. WHEN incoming_prod == 'many' AND free_gpus >= 4:
    reserve 2 GPUs
    [learned: Reserving 2 GPUs specifically when incoming_prod is 'many' ensures better SLA protection without over-reserving when there are fewer prod jobs.]

P2. WHEN incoming_prod != 'none' AND (free_gpus == 2 OR free_gpus == 3):
    reserve 1 GPU
    [learned: Reserving 1 GPU in these cases balances future protection with higher utilization, as seen in this window.]

P3. WHEN incoming_prod == 'many' AND free_gpus >= 6:
    reserve 3 GPUs
    [learned: Reserving more GPUs when many prod jobs are incoming and capacity is high improves their SLA without significantly affecting utilization.]

P4. WHEN incoming_prod == 'many' AND free_gpus == 5:
    reserve 3 GPUs
    [learned: Increasing the reserve to 3 GPUs improves SLA protection for prod jobs without significantly affecting utilization, as seen in this window.]

P5. WHEN incoming_prod == 'few' AND (free_gpus >= 2 AND free_gpus <= 4):
    reserve 1 GPU
    [learned: Reserving 1 GPU when incoming_prod is 'few' and free_gpus is between 2-4 improves SLA without over-reserving, as seen in this window.]

P6. WHEN incoming_prod == 'many' AND free_gpus >= 7:
    reserve 3 GPUs
    [learned: Reducing the reserve to 3 GPUs when many prod jobs are incoming and capacity is high improves utilization while still protecting SLA, as seen in this window.]

P7. WHEN incoming_prod == 'many' AND free_gpus == 4:
    reserve 3 GPUs
    [learned: Reserving 3 GPUs when incoming_prod is 'many' and free_gpus is exactly 4 improves SLA protection for prod jobs without significantly affecting utilization.]

P8. WHEN incoming_prod == 'many' AND free_gpus >= 5:
    reserve 3 GPUs
    [learned: Consistently reserving 3 GPUs when many prod jobs are incoming and capacity is sufficient improves SLA protection without significantly affecting utilization, as seen in this window.]
