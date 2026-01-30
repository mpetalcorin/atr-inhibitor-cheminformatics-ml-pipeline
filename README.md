# atr-inhibitor-cheminformatics-ml-pipeline
Reproducible end-to-end pipeline for ATR inhibitor discovery, from BindingDB curation and drug-likeness filtering to ML benchmarking, interpretability, and ranked candidate shortlists.

# ATR Inhibitor Discovery, Cheminformatics + ML Pipeline (BindingDB → Models → Ranked Candidates)

This repository is an end-to-end, reproducible cheminformatics and machine-learning workflow for **ATR (ataxia telangiectasia and Rad3 related) kinase** inhibitor discovery. It takes raw BindingDB bioactivity records, cleans and harmonises activity labels, applies **drug-likeness filtering**, trains and compares multiple ML models, explains predictions with **Shapley-based interpretability**, and produces **ranked candidate shortlists** suitable for experimental follow-up.

## What this does
1. **Collect** ATR activity data (SMILES + IC50/Ki/Kd) from BindingDB  
2. **Clean** the dataset (units, duplicates, missing values) and define **Active vs Inactive** labels  
3. **Filter** to more realistic small molecules (drug-likeness rules)  
4. **Represent** each molecule as numbers (fingerprints, descriptors)  
5. **Train and evaluate** multiple ML models (scikit-learn, optionally PyTorch)  
6. **Explain** what chemical features drive predictions (Shapley-style explanations)  
7. **Generate/score** new candidate molecules (SELFIES/ChemGPT-style generation) and output **top-ranked lists**

## Repository contents
- `ATR Drug Discovery_ML.ipynb`  
  Main notebook, runs the full pipeline from preprocessing to model training, evaluation, interpretability, and candidate ranking.

Recommended folder layout:
```
├── ATR Drug Discovery_ML.ipynb
├── data/
│   ├── raw/
│   │   └── bindingDB_ATR.tsv
│   └── processed/
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── candidates/
└── README.md
```
## Inputs
### Required
- `data/raw/bindingDB_ATR.tsv`  
  BindingDB export for ATR containing SMILES and activity fields (IC50, Ki, Kd).

### Optional
- Any additional ATR-focused assay tables you want to merge (for example internal assay exports). If you add extra sources, keep a provenance column so the pipeline remains auditable.

## Outputs (what you get)
Depending on which cells you execute, you will generate files similar to:

**Processed datasets**
- `ATR_preprocessed.csv`, `ATR_preprocessed_renamed.csv`  
  Cleaned bioactivity table with consistent columns and labels.
- `ATR_inhibitors_druglike_filtered.csv`  
  Filtered “more realistic” small molecules.

**Model artefacts and benchmarking**
- `model_performance_summary.csv`  
  Accuracy, ROC-AUC, precision, recall, and other metrics per model.
- `roc_curves_atr_models.png`  
  Visual comparison of model performance.

**Interpretability**
- SHAP/feature importance plots and tables explaining which molecular descriptors contributed most to predictions.

**Ranked candidates**
- `chemGPTselfies_ATR_BDcandidates.csv`  
  Generated molecules scored by the trained model.
- `top10_ATR_candidates_named.csv` (or similar)  
  A compact shortlist for review and procurement/synthesis planning.

## Quickstart
### 1) Create a Python environment
Use conda or venv, any modern Python 3 works.

### 2) Install dependencies
Typical requirements:
- `rdkit`
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`
- `imbalanced-learn` (for SMOTE-like balancing)
- `mordred` (if you use Mordred descriptors)
- `shap`

### 3) Add data
Place your BindingDB export at:
`data/raw/bindingDB_ATR.tsv`

### 4) Run the notebook
Open and run:
`ATR Drug Discovery_ML.ipynb`

Tip: run from top to bottom the first time, then rerun only the sections you are iterating on (model selection, thresholds, candidate generation).

## Key design choices (why the pipeline is credible)
- **Data provenance and standardisation**: chemical structure standardisation and curation are treated as first-class steps to avoid “garbage in, garbage out” behaviour.  
- **Imbalanced learning**: class imbalance is addressed with oversampling strategies to reduce bias toward the majority class, while recognising risks of synthetic data and validating carefully. 
- **Interpretability**: Shapley-style explanations are used to make model behaviour understandable and chemically reviewable, supporting better decisions.
- **Robust molecular strings for generation**: SELFIES-based generation reduces invalid molecule strings and improves generative workflow stability.

## How to use the results responsibly
This repository produces **prioritised hypotheses**, not confirmed inhibitors. Use the ranked lists to:
- select a manageable set of compounds for **purchase, synthesis, and biochemical assays**
- check liabilities and novelty
- iterate using experimental feedback (active learning style)

## Suggested extensions
- Add scaffold-aware splitting and applicability-domain checks for more realistic generalisation.
- Add SHAP-based chemical sanity checks to detect spurious descriptor reliance.
- Add multi-objective ranking (potency + ADMET proxies + synthetic accessibility).
- Add docking or structure-based scoring as a complementary filter (SBDD interface).

## References
- Bento, A. P., Hersey, A., Félix, E., Landrum, G., Gaulton, A., Atkinson, F., Bellis, L. J., De Veij, M., & Leach, A. R. (2020). An open source chemical structure curation pipeline using RDKit. Journal of Cheminformatics, 12, 51. https://doi.org/10.1186/s13321-020-00456-1
- Rodríguez-Pérez, R., & Bajorath, J. (2020). Interpretation of machine learning models using Shapley values, application to compound potency and multi-target activity predictions. Journal of Computer-Aided Molecular Design, 34(10), 1013–1026. https://doi.org/10.1007/s10822-020-00314-0
- Krenn, M., Häse, F., Nigam, A., Friederich, P., & Aspuru-Guzik, A. (2022). SELFIES and the future of molecular string representations. Patterns, 3(10), 100588. https://doi.org/10.1016/j.patter.2022.100588
- Nakamura, M., Kajiwara, Y., Otsuka, A., & Kimura, H. (2013). LVQ-SMOTE, Learning Vector Quantization based Synthetic Minority Over-sampling Technique for biomedical data. BioData Mining, 6, 16. https://doi.org/10.1186/1756-0381-

## License
MIT; CC BY for documentation

# Model Card, ATR Inhibitor Classifier (Cheminformatics + ML)

## Model details
**Model name:** ATR Inhibitor Classifier  
**Model type:** Binary classifier (Active vs Inactive) for ATR bioactivity  
**Intended use:** Screening prioritisation, rank-ordering, and hypothesis generation for ATR (ataxia telangiectasia and Rad3 related kinase) inhibitor discovery.  
**Primary users:** Cheminformatics scientists, ML scientists, medicinal chemists, computational biologists.  
**Version:** v1.0 (repository release)  
**Owner/Maintainer:** Mark Ihrwell R. Petalcorin, PhD  

## Summary
This model predicts whether a small molecule is likely to be **active** against **ATR** based on its chemical structure. The model is trained on curated **BindingDB** ATR records, where activity measurements (IC50, Ki, Kd) are harmonised and converted into a binary label using a predefined threshold. Outputs are intended to support **compound triage** and **shortlist generation** for experimental follow-up, not to replace biochemical validation.

## Model inputs and outputs
### Inputs
- **SMILES** strings representing small-molecule structures.
- The pipeline converts SMILES into machine-readable features using one or more of:
  - **RDKit fingerprints** (e.g., Morgan/ECFP)
  - **Physicochemical descriptors**
  - **Mordred descriptors** (if enabled)

### Outputs
- **Predicted class:** Active (1) or Inactive (0)  
- **Score/probability:** Model confidence for the Active class (when probability-calibrated models are used)  
- Optional:
  - **Uncertainty estimates** (e.g., from ensembles)
  - **Interpretability artefacts** (e.g., SHAP summary)

## Training data
### Data source
- **BindingDB** export for ATR, containing SMILES and assay measurements (IC50, Ki, Kd).

### Label definition
- Activity is harmonised by selecting a single activity value per record using a priority rule (e.g., IC50 then Ki then Kd), converting to a consistent unit (nM), and mapping to:
  - **Active (1):** activity < 1000 nM  
  - **Inactive (0):** activity ≥ 1000 nM  
(Threshold may be adjusted; changes should be documented per release.)

### Preprocessing
Typical preprocessing includes:
- Removing missing/invalid SMILES
- Deduplication (e.g., by SMILES or InChIKey)
- Optional drug-likeness filtering (Lipinski-style constraints)
- Handling class imbalance (e.g., SMOTE) when appropriate

### Dataset limitations
- BindingDB aggregates results from different assay formats, conditions, and labs, so labels may include measurement noise and assay heterogeneity.
- If oversampling (SMOTE) is used, the training distribution may include synthetic samples and requires careful validation.

## Evaluation
### Metrics
Common metrics reported in this repository include:
- ROC-AUC
- Accuracy
- Precision, recall
- Confusion matrix

### Validation strategy
- Standard random splits may overestimate performance if chemically similar series leak across splits.
- More realistic evaluation may include **scaffold-aware splitting**; if applied, results should be reported alongside random split results.

### What “good performance” means here
A strong ROC-AUC or accuracy indicates good separation on the chosen split, but the most important measure for real use is whether the top-ranked compounds are enriched for true actives in prospective testing.

## Intended use
### Primary intended uses
- Ranking compounds for **purchase/synthesis**
- Enriching screening sets for likely ATR inhibition
- Supporting iterative design cycles with medicinal chemistry teams
- Generating shortlists for downstream validation

### Out-of-scope uses
- Clinical decision-making
- Safety/toxicity prediction
- ADMET or PK prediction
- Claims of ATR selectivity versus other kinases without additional evidence
- Use in regulated contexts without validation and governance

## Risks and limitations
- **Assay heterogeneity:** Different assay types can produce different apparent potencies.
- **Distribution shift:** Novel chemical series may be out-of-domain relative to training data.
- **Bias toward training chemistry:** The model may favour scaffolds common in the dataset.
- **Imbalance handling:** Synthetic oversampling can inflate apparent performance if validation is not strict.
- **Interpretability caveats:** SHAP/feature importance can highlight correlates, not causal mechanisms.

## Fairness and bias considerations
This model does not operate on human subjects or demographic attributes. However, chemical bias can occur:
- Over-representation of specific scaffolds, suppliers, or chemotypes in BindingDB.
- Under-representation of novel or proprietary chemical space.
Mitigation includes scaffold-aware splits, applicability-domain checks, and prospective validation.

## Explainability
The repository may include **SHAP** analyses to help interpret which molecular descriptors contribute to predictions. Explainability outputs should be reviewed by domain experts for chemical plausibility.

## Environmental impact
Training and evaluation are designed to run on standard CPU hardware. If deep learning models or large descriptor sets are used, compute demand increases. Prefer reusing cached descriptors and keeping runs reproducible to minimise repeated compute.

## Reproducibility
To reproduce results:
1. Use the repository notebook `ATR Drug Discovery_ML.ipynb`
2. Ensure dependency versions match your environment
3. Place the BindingDB ATR export in the expected location (see README)
4. Run the notebook from top to bottom

## Maintenance
- Report issues via GitHub Issues.
- If updating thresholds, feature sets, or split strategies, bump the model version and record changes in a changelog.

**Disclaimer:** This model generates computational hypotheses. Experimental validation is required before drawing conclusions about ATR inhibition or suitability for drug development.

# Datasheet for Dataset, ATR Bioactivity (BindingDB → Curated → Model-Ready)

## 1. Dataset name
**ATR Bioactivity Dataset (BindingDB-curated)**

## 2. Dataset summary
This dataset contains small molecules annotated with **ATR (ataxia telangiectasia and Rad3 related kinase) bioactivity** measurements extracted from **BindingDB** and processed into a curated, model-ready format for cheminformatics and machine learning.

The dataset supports:
- Binary classification, **Active vs Inactive** ATR inhibition
- Benchmarking ML models on curated chemical + bioactivity tables
- Producing ranked candidate lists for experimental follow-up

## 3. Motivation
Public bioactivity data for kinase targets is often heterogeneous, noisy, and difficult to use directly for ML. This dataset was created to:
- harmonise activity fields (IC50, Ki, Kd),
- standardise units,
- reduce duplicates,
- add simple drug-likeness filtering,
- and provide a reproducible baseline for ATR inhibitor prediction.

## 4. Composition
### 4.1 What are the instances
Each row represents a **compound–ATR activity record** with:
- a molecular structure string (SMILES),
- a selected activity value (in nM),
- and a derived binary activity label.

### 4.2 Data fields (typical columns)
Depending on pipeline settings, the dataset may include:
- `SMILES` (string): chemical structure
- `activity_nM` (float): chosen potency value in nM
- `activity_type` (string): which measurement was used (IC50, Ki, or Kd)
- `class` (int): binary label (1 = Active, 0 = Inactive)
- Optional chemistry properties (if computed):
  - molecular weight, logP, HBD/HBA, TPSA, rotatable bonds
- Optional ML features (if precomputed and stored):
  - RDKit fingerprints
  - Mordred descriptors

### 4.3 Size and splits
- Dataset size depends on the BindingDB export date and filtering steps.
- Train/test splits may be random or scaffold-based depending on the notebook configuration.

## 5. Data collection process
### 5.1 Source
- **BindingDB** ATR target export (SMILES + activity values).

### 5.2 How it was collected
- Exported BindingDB table for ATR was ingested as a TSV.
- Activity measurements were read from available fields (IC50, Ki, Kd).
- Records missing both structure and activity were removed.

### 5.3 Time period
- Depends on the BindingDB export used in the repository.  
Document the export date in the repository (recommended).

## 6. Data processing and cleaning
### 6.1 Activity harmonisation
- A single activity value is selected per record using a priority rule:
  - Prefer IC50, otherwise Ki, otherwise Kd (if present).
- Values are converted into a consistent unit (**nM**) and parsed as numeric.

### 6.2 Deduplication
- Duplicate structures are removed (commonly by SMILES; ideally by InChIKey).
- If duplicates have conflicting activity labels, the conflict should be logged and resolved consistently (e.g., keep best potency, median, or drop conflicts).

### 6.3 Label generation
A binary label is derived by thresholding potency:
- **Active (1):** activity < 1000 nM  
- **Inactive (0):** activity ≥ 1000 nM  
(If the threshold is changed, record the new threshold and rationale.)

### 6.4 Drug-likeness filtering 
A “drug-like” subset is generated using simple physicochemical constraints (Lipinski-style filtering).
This reduces unrealistic chemistry and improves relevance for discovery.

### 6.5 Class imbalance handling 
If the Active class is underrepresented, oversampling may be applied (e.g., SMOTE).
This creates a separate “balanced” dataset and should be treated as a modelling convenience, not as ground truth expansion.

## 7. Uses
### 7.1 Intended uses
- Train baseline ML classifiers for ATR inhibition
- Compare algorithms and featurisation strategies
- Generate ranked shortlists for **experimental validation**
- Demonstrate reproducible cheminformatics + ML workflows

### 7.2 Out-of-scope uses
- Clinical decision-making
- Toxicity, ADMET, or PK prediction
- Claims of ATR selectivity against other kinases
- Regulatory or safety-critical use without additional validation and governance

## 8. Distribution shift and limitations
### 8.1 Assay heterogeneity
BindingDB aggregates measurements from different assay conditions, labs, and endpoints.
This can introduce label noise and apparent contradictions.

### 8.2 Chemical coverage bias
The dataset reflects what has been tested and deposited, which may overrepresent certain scaffolds or “popular” chemistry.

### 8.3 Split leakage risk
Random splits can inflate performance because similar chemotypes appear in both train and test sets.
Scaffold-based splitting is recommended for more realistic evaluation.

### 8.4 Oversampling risk
If SMOTE is used, synthetic samples can lead to overly optimistic metrics if evaluation is not strict.

## 9. Ethical considerations
This dataset contains **no human subjects data** or personally identifiable information.
Ethical considerations mainly relate to:
- responsible interpretation of predictions as hypotheses,
- avoiding overconfidence without experimental confirmation,
- and respecting BindingDB data usage terms.

## 10. Licensing and access
- BindingDB data is publicly accessible; users must comply with BindingDB terms and citation guidance.
- This repository distributes **derived** processed tables; confirm license compatibility for redistribution.

## 11. Recommended citations 
- BindingDB, a public database of measured binding affinities. *Nucleic Acids Research*.  
- Bento AP, et al. An open source chemical structure curation pipeline using RDKit. *J Cheminform*. 2020.  
- Rodríguez-Pérez R, Bajorath J. Interpretation of machine learning models using Shapley values. *J Chem Inf Model*. 2020.

## 12. Contacts
**Maintainer:** Mark I.R. Petalcorin 
**Email:** m.petalcorin@gmail.com  
**GitHub:** github.com/mpetalcorin
**Disclaimer:** This dataset supports computational screening and model development. Experimental validation is required before concluding ATR inhibition or suitability for drug development.
