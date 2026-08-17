# ASL_PerfEPP10+ — Cerebral Blood Flow in Schizophrenia (PerfEPP10+)

## Project Overview

This project investigates regional cerebral blood flow (rCBF) patterns in patients with
schizophrenia using arterial spin labelling (ASL) MRI, as part of the PerfEPP10+ study.
Patients are former PEPP participants with a DSM-5 diagnosis of schizophrenia. The study
examines perfusion differences and network-level connectivity between schizophrenia patients
and healthy controls.

Arterial spin labelling (ASL) is a non-invasive MRI technique that uses magnetically 
labelled water in arterial blood as an endogenous tracer to measure cerebral blood flow 
(CBF). Unlike PET or SPECT, ASL requires no exogenous contrast agent, making it 
particularly suited for clinical and research applications. CBF measured with ASL 
reflects the volume of blood delivered to brain tissue per unit time (ml/100g/min) and 
serves as a proxy for neural activity and neurovascular health.

**Institution:** Douglas Neuroinformatics Platform (DNP), Douglas Mental Health University Institute  
**PI:** Delphine Raucher-Chene
**Data collection by:** Olivier Percie du Sert, Delphine Raucher-Chene, Thomas 
**Analysis done by:**  Rubana Murshed
**Last updated:** March 2026

---

## Repository Structure
```
ASL_PerfEPP10+/
├── README.md
├── requirements.txt
├── code/
│   ├── preprocessing/
│   │   ├── aslprep_rubana.sh        ← main ASLPrep job script (LSF/cluster)
│   │   └── lsfg_4028.sh             ← single-subject job script for sub-4028
│   └── analysis/
│       ├── cbf_correlation_analysis.py  ← CBF group-level ROI correlations
│       ├── smallworld.py                ← small-world network analysis
│       ├── node_permutation.py          ← permutation testing for node metrics
│       ├── check_nodes.py               ← QC/inspection of network nodes
│       ├── final_visualization.py       ← generates all final figures
│       └── graph_cbf/
│           ├── make_4S256_cbf_timeseries.py  ← builds CBF timeseries for graph analysis
│           └── run_cbf_graph_pipeline.py     ← runs full CBF graph construction pipeline
├── containers/
│   └── aslprep-25.1.0.sif           ← Singularity image for ASLPrep v25.1.0
├── config/                          ← ASLPrep configuration files
├── rawdata/                         ← original raw data (read-only, do not modify)
├── rawdata_clean/                   ← cleaned BIDS data actually fed into ASLPrep
│   └── sub-XXXX/ses-01/
│       ├── anat/
│       ├── func/
│       └── perf/                    ← ASL + M0 scan NIfTIs and JSON sidecars
├── derivatives/
│   └── aslprep/                     ← ASLPrep outputs (BIDS derivatives)
│       ├── sub-XXXX/                ← per-subject CBF maps and outputs
│       ├── sub-XXXX.html            ← per-subject visual QC reports (open in browser)
│       ├── atlases/
│       ├── logs/
│       └── sourcedata/
├── results/
│   ├── figures/
│   │   ├── final_results.png
│   │   ├── network_analysis_results.png
│   │   ├── network_level_summary.png
│   │   ├── gm_cbf_by_group_centered.png
│   │   ├── lmfg_cbf_by_group_centered.png
│   │   ├── lput_cbf_by_group_centered.png
│   │   ├── lsfg_cbf_by_group_centered.png
│   │   └── rmog_cbf_by_group_centered.png
│   ├── stats/
│   │   ├── averageGMCBFcontrols.csv
│   │   ├── averageGMCBFpatients.csv
│   │   ├── group_gm_cbf_means.csv
│   │   ├── group_gm_cbf_means_with_group_updated.csv
│   │   ├── left_middle_frontal_gyrus.tsv
│   │   ├── left_putamen.tsv
│   │   ├── left_superior_frontal_gyrus.tsv
│   │   ├── right_middle_occipital_gyrus.tsv
│   │   ├── correlation_results/
│   │   └── graph_cbf/
│   │       ├── allSubs_4S256_cbf_graphMerged.tsv
│   │       └── allSubs_4S256_globalMetrics.tsv
│   └── matrices/
│       ├── control_matrices.npy
│       ├── schiz_matrices.npy
│       ├── node_degree_diff.npy
│       ├── node_pvalues.npy
│       ├── node_pvalues_fdr.npy
│       └── cbf_vectors/
└── docs/
    └── PerfEPP10_description.pdf
```

---

## Participants

| Group    | Subject IDs                                                                               | N |
|----------|-------------------------------------------------------------------------------------------|---|
| Patients | sub-4028, sub-4042, sub-4045, sub-4046, sub-4049, sub-4061, sub-4062, sub-4063, sub-4067 | 9 |
| Controls | sub-5002, sub-5008, sub-5009, sub-5012, sub-5014, sub-5016                                | 6 |

---

## How to Reproduce the Analysis

### Prerequisites

- **Cluster access:** DNP (DNPWS12), LSF job scheduler
- **Singularity:** Required to run ASLPrep
- **Python:** >= 3.8, with dependencies in `requirements.txt`
```bash
python3 -m venv asl_analysis
source asl_analysis/bin/activate
pip install -r requirements.txt
```

### Step 1 — ASL Preprocessing (ASLPrep)

ASLPrep v25.1.0 was used to preprocess the cleaned BIDS data and generate CBF maps.
```bash
bsub < code/preprocessing/aslprep_rubana.sh
```

- Input: `rawdata_clean/`
- Output: `derivatives/aslprep/`
- Container: `containers/aslprep-25.1.0.sif`
- Config: `config/`

> Check per-subject QC HTML reports in `derivatives/aslprep/sub-XXXX.html` before proceeding.

### Step 2 — CBF Graph Construction
```bash
python code/analysis/graph_cbf/make_4S256_cbf_timeseries.py
python code/analysis/graph_cbf/run_cbf_graph_pipeline.py
```

- Input: `derivatives/aslprep/`
- Output: `results/matrices/`, `results/stats/graph_cbf/`

### Step 3 — CBF Group Statistics and ROI Analysis
```bash
python code/analysis/cbf_correlation_analysis.py
```

- Input: `derivatives/aslprep/`
- Output: `results/stats/`, `results/figures/`

### Step 4 — Network / Graph Theory Analysis
```bash
python code/analysis/smallworld.py
python code/analysis/node_permutation.py
python code/analysis/check_nodes.py
python code/analysis/final_visualization.py
```

- Input: `results/matrices/`
- Output: `results/figures/`, `results/stats/`

---

## Key Outputs

| File | Description |
|------|-------------|
| `results/figures/gm_cbf_by_group_centered.png` | Global GM CBF by group |
| `results/figures/network_analysis_results.png` | Network-level group differences |
| `results/matrices/node_pvalues_fdr.npy` | FDR-corrected node p-values (main result) |
| `results/stats/graph_cbf/allSubs_4S256_globalMetrics.tsv` | Global graph metrics per subject |
| `results/stats/group_gm_cbf_means_with_group_updated.csv` | Final CBF summary table |

---

## Environment

- **Platform:** DNP (Douglas Neuroinformatics Platform), SSH remote (DNPWS12)
- **Scheduler:** LSF (`bsub`)
- **ASLPrep version:** 25.1.0 (Singularity container)
- **Python packages:** see `requirements.txt`

---

## Notes

- `rawdata/` = original files from scanner; `rawdata_clean/` = cleaned BIDS version used for analysis
- Working directories (`aslwd/`, `asltmp/`) were not archived — derivatives confirmed complete before deletion

---

## Contact

**Rubana Murshed**  
rubana.mursh@gmail.com
Douglas Mental Health University Institute
