# Machine-learning imputation for ECB SAFE Q8A

**TL;DR**
CatBoost-based imputation of ECB/EC SAFE Q8A (loan-size brackets) to reconstruct the loan-size component used in firm-level bank financing-gap estimation, without external covariates.

## Methodology
This repository contains the research code for my MSc thesis “Improving granular financing gap estimates using Machine Learning and Small Area Estimation - Evidence from EU countries” (Bocconi University, supervisor: Prof. Francesco Corielli). The main goal is to recover missing values for SAFE question Q8A (requested/obtained bank-loan size, recorded in brackets), avoiding external data sources to preserve clean downstream econometric interpretation (i.e., minimising endogeneity concerns from the imputation stage). More information and results can therefore be found in the full research paper.

The code implements two final CatBoost imputation specifications:
- Model I: CatBoost MultiClass on Q8A bins + probabilistic imputation
- Model II: CatBoost regression on Q8A midpoint + snap back to SAFE midpoints

Q8A was discontinued from 2023 onward (structural missingness) and also exhibits item non-response in earlier waves. This pipeline reconstructs a consistent loan-size component for financing-gap estimation over time. **The core CatBoost imputation approach was developed by me and implemented in the European Commission JPP final report *“Firm-Level Financing Gap in the Eurozone”*;** the thesis version in this repository further refines and extends that methodology (e.g., additional diagnostics and model variants). 

## Note on data access
ECB/EC SAFE microdata are not redistributed here. The repository is structured so that authorised users can plug in the SAFE extract locally and reproduce the full pipeline.
