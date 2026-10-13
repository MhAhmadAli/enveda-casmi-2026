# CASMI26 simulated ranker rows

Training rows for a candidate ranker in the Enveda CASMI 2026 competition.

- 2,250 simulated molecules from 7 query sets (enveda-np-examples 250, gnps 400, riken 300, mona 300, msdial 300, massbank 300, pluskal_ms2 400)
- Each molecule appears twice: `S=0` class-1 simulation (query library removed), `S=1` class-2 simulation (all spectra of the true structure removed, structure kept in the pool)
- Candidate pool: COCONUT (prvsiyan/coconut-casmi26-candidates) + training structures, +-10 ppm
- `sim_rank_rows_nofp.npz`: 399,790 rows x 51 features (`blocks` array names the feature blocks), `Y` = true candidate, `G` = group, `Q` = query set
- No structures, SMILES or spectra are included; only numeric features.

## v3 additions
- `ours_fpnet_single_16k.pt`: prvsiyan's FPNet architecture retrained from scratch on a T4 (20k steps, best validation at 16k, hardneg-top1 0.436) with **all 3,150 simulation structures held out**. Single-channel Class-2 MRR on enveda-np-examples 0.452 (public model 0.493); on the other six query sets 0.52-0.62 (public model 0.76-0.82, because it had seen those structures).
- `sim_rank_rows_fp16k.npz`: 399,790 rows x 39 features (`base` + `libx` blocks) whose fingerprint features come from the leak-free model above. 5-fold grouped CV on enveda-np-examples: Class-2 MRR 0.655 -> 0.689 versus the no-fingerprint rows.

## v4 additions
- `ours_fpnet_merged_26k.pt`: the merged-spectrum view of the same leak-free FPNet (26k steps on a T4, best validation 0.476 vs 0.436 for the per-spectrum model).
- `sim_rank_rows_fppair.npz`: ranker rows whose fingerprint features come from **both** leak-free models (per-spectrum + merged, averaged views). Single-channel Class-2 MRR on enveda-np-examples 0.478 (public model 0.493, but that one has seen most library structures); our ranker's Class-2 MRR 0.689 -> 0.710.
