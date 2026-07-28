# Referee manual — SELF-AUTHORED (Exp 51 Phase A; do not hand-edit, regenerate via pins/manual_author.py)

<!-- PROMPT-START: everything below this line is sent to the referee -->
PRECEDENTS from past allocations (apply when the WHEN clause matches; cite the precedent id in your justification):

P1. WHEN incoming_prod != 'none' AND free_gpus >= 2:
    prioritize allocating GPUs to meet base requirements of all incoming prod jobs before considering margins for besteffort jobs
    [learned: Ensuring base needs met improved SLA by reducing deadline violations.]

P2. WHEN incoming_prod == 'many' AND free_gpus >= 2:
    if free_gpus > 4 then reserve exactly 3 GPUs, else if free_gpus > 2 then reserve exactly 2 GPUs, else reserve all available GPUs to protect incoming prod jobs while allowing margin allocation for besteffort jobs if possible
    [learned: Increasing reservation when free_gpus is high improved protection for critical prod jobs without over-reserving.]

P3. WHEN incoming_prod == 'many' AND free_gpus <= 2:
    reserve exactly 1 GPU to protect incoming prod jobs while allowing margin allocation for besteffort jobs if possible
    [learned: Reducing the reservation from 2 to 1 GPU when free_gpus is limited improves besteffort job utilization without compromising critical prod protection.]

P4. WHEN incoming_prod != 'none' AND free_gpus > 2:
    prioritize allocating margin GPUs to besteffort jobs after reserving exactly 2 GPUs for incoming prod jobs, ensuring their deadlines are met where possible
    [learned: This adjustment reduces besteffort deadline violations while maintaining adequate protection for critical prod jobs.]

P5. WHEN incoming_prod == 'few' AND free_gpus > 2:
    reserve exactly 1 GPU to protect incoming prod jobs while allowing margin allocation for besteffort jobs if possible
    [learned: Reducing the reservation when free_gpus is exactly 2 improved besteffort job utilization without compromising critical prod protection.]

P6. WHEN free_gpus >= 1 AND free_gpus <= 3:
    reserve exactly 1 GPU to protect incoming prod jobs while allowing margin allocation for besteffort jobs if possible
    [learned: Balancing protection and utilization reduced deadline violations across both tiers.]
