# Debate v2.4.3 sweep override

The v2.4.3 pilot gate remains frozen and will still be reported as pass or fail. Before pilot
completion, the user explicitly authorized running the 24-window training sweep regardless of that
gate because the results are needed for a presentation the next day.

The already-running `d111h22` and subsequent `d223h11` pilot use exactly the sweep protocol, model,
seed, simulator interval, selector interval, family, sizing start, and fallback settings. Their rows
are therefore counted in the 24-window result with `source_tag=pilot_v243_s17`; the sweep runner
executes the other 22 windows under `source_tag=sw_v243_s17`. This avoids two redundant long runs
while preserving one result for every frozen training window.
