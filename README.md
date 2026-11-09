# When a Clear Is Not a Payment: code and results

Code for the analyses in *When a Clear Is Not a Payment: A Mixed-Methods Analysis of Control Deviations and Payment Time in an SAP Purchase-to-Pay Event Log* (X.-A. Tran, P. Nguyen), and for the undergraduate thesis it is based on (University of Economics and Law, VNU-HCM). Script names, comments and printed output are in Vietnamese; this file maps each step to the paper. A Vietnamese README with more detail is in [`README_vi.md`](README_vi.md).

## Data

BPI Challenge 2019 event log (van Dongen, 2019), DOI [10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1). The log (about 700 MB) is not included. Download `BPI_Challenge_2019.xes` and place it in `du_lieu/`, or set the environment variable `KLTN_LOG` to its path, or pass the path as the first argument of a script.

## Installation

```
pip install -r requirements.txt
```

Library versions used for the reported results are listed in `requirements_da_chay.txt`. Graphviz is optional (only for PNG process maps).

## Running

Run every command from this folder, in the order below; some scripts reuse functions or outputs of earlier steps. Scripts with `--tu-kiem` run unit tests on synthetic traces first (recommended). Each script writes to its own `ket_qua_*` folder.

| Step | Command | What it does | Paper |
|---:|---|---|---|
| 1 | `python src/kiem_tra_trang_thai_case.py` | Reads the log; activities; extraction time; edge cases | §3.1 |
| 2 | `python src/bang3_pheu_loc.py` | Scope filter (251,734 → 210,887 items) | §3.1 |
| 3 | `python src/r5_v2_phan_loai_clear.py --delta-phut 60 --ra ket_qua_r5_v2_d60` | Separates payment clearing from reversal clearing | §3.2 |
| 4 | `python src/g3_kiem_tra_bo_sung.py` | Choice of Δ; timestamp granularity | §3.2 |
| 5 | `python src/g4_kiem_tra.py` | R5, debit memos, R4 merged into R5; time zone; rework indicator | §3.2, §3.4 |
| 6 | `python src/g5_kiem_tra.py` | Line-item status (completed / open / deleted) | §3.2 |
| 7 | `python src/kiem_chung_cuoi.py` | Rule set R1–R7b; validation samples (seed 2026) | §3.4, §4.2 |
| 8 | `python src/dem_su_kien_trang_thai.py` | Items and events by status | §3.2 |
| 9 | `python src/sq1_kham_pha.py` | Activity frequencies, variants, branches, batch accounts; normative Petri net (`.pnml`) and its tests | §3.3, §4.1 |
| 10 | `python src/sq1_imf.py` | Inductive Miner – infrequent, compared with the normative model | §4.1 |
| 11 | `python src/sq2_tuan_thu.py` | Rule prevalence, concentration by vendor and category; alignments | §4.1, §4.2 |
| 12 | `python src/sq2_bo_sung.py` | Additional rule analyses (R6 groups, R1 on items without cancellations) | §4.2 |
| 13 | `python src/kiem_chung_bo_sung.py` | Joins manual reading with rule output; R3 mechanism; alignments on all variants | §3.4, §4.1, §4.2 |
| 14 | `jupyter notebook 02_kham_pha.ipynb` | Reproduces tables and figures of steps 9–13 | §4.1 |
| 15 | `python src/sq3_thoi_gian.py` | Kaplan–Meier; Cox with time-varying covariates; day-30 landmark; Schoenfeld; RMST | §3.5, §4.3 |
| 16 | `python src/sq3_kiem_tra_co_che.py` | From signal to payment; rework and status definition | §4.3 |
| 17 | `jupyter notebook 04_thoi_gian.ipynb` | Reproduces steps 15–16 | §4.3 |
| 18 | `python src/g8_kiem_tra.py` | Sensitivity analyses: vendor-stratified Cox, R6 without two vendors, R7b from a common time origin, creation-time covariates | §4.3, Fig. 3 |
| 19–21 | `src/sq4_mo_ta.py`, `src/sq4_mo_hinh.py`, `src/sq4_phan_tich_sau.py` | Early prediction of deviations (thesis only, not in the paper) | – |

Steps 10–11 need PM4Py, step 13 also SciPy, and steps 15–18 lifelines. Step 13 is the slowest because it aligns every variant (`--bo-alignment` skips this part).

## Results included

The `ket_qua_*` folders contain summary files (`tom_tat_*.txt`), small tables, the normative model (`.pnml`), the validation samples and the manual-reading files (`*_nhan_xet.csv`). Item-level output files larger than 2 MB are not included and are recreated by running the scripts. The validation samples were drawn on 18 October 2026 with seed 2026; `kiem_chung_bo_sung.py` redraws them with the same code and seed and joins them with the manual-reading files.

## Citation

Please cite the paper and the dataset (van Dongen, B.F.: BPI Challenge 2019. 4TU.ResearchData (2019). https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1).
