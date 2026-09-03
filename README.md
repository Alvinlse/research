# results-archive (orphan branch)

Compressed snapshots of the large `pins/results_*.json` files that are git-ignored on the
working branches (each >100 MB uncompressed; ~60x smaller with zstd). Archived 2026-09-03 to
free space on the login node. `results/MANIFEST.sha256` holds the sha256 of every ORIGINAL
(uncompressed) file.

Restore one file into a checkout of `referee_allocator`:

    git show results-archive:results/results_backup_pre_exp84_debate.json.zst | zstd -d > pins/results_backup_pre_exp84_debate.json
    sha256sum -c <(grep results_backup_pre_exp84_debate.json <(git show results-archive:results/MANIFEST.sha256)) --ignore-missing

What each file is:
- `results_backup_pre_<exp>.json` — snapshot of `pins/results_trace_replay.json` taken before that experiment
  (Exp 49-59, 84, 87, phase B) — rows later overwritten by --extend live only here.
- `results_trace_replay.json` — the live results file as of 2026-07-29 (still present locally).
- `results_trace_replay.json.corrupt_bak` — the clobbered copy from 2026-07-22 (JSON clobber trap).
- `results_trace_replay.pre-*.bak.json` — older small snapshots (Exp 43 era).
- `results_exp99.json` — Exp 99 results (stripped from branch history 2026-08-04).
