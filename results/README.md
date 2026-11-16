# Results of the paper

The numbers behind the tables and figures of the revised paper, as written by the
scripts in `scripts/` with the models in `checkpoints/`.

| folder | paper | written by |
|---|---|---|
| `final_check/<system>_summary.md` | verified regions (Sec. VII-A) and Table II | `region_check.py`, `prediction_benchmark.py`, `collect_results.py` |
| `prediction/<system>_fresh.json` | Table II, Fig. 7 | `prediction_benchmark.py <system>` |
| `ablation_dictionary/<system>_region/seed_*/results.json` | Figs. 5-6 and the ablation B text | `ablation_dictionary.py` |
| `control_comparison/<system>_<method>_seed<seed>.json` | Table III, Fig. 8 | `control_comparison.py` |

In `final_check`, the region check is an independent draw of 1500 parameters from
each verified region (seed 2031). The manipulator region was built and checked at
20%. The double-pendulum NMPC entry of Table III (25.631 s) comes from the runs of
the first preprint; its script was not kept.
