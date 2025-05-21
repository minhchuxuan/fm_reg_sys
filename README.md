

# Multi-Field CTR Prediction: An Evolution of Factorization Machines (IT3190E)

This repository contains the materials and code related to our Machine Learning (IT3190E) course project, focusing on the evolution of Factorization Machines for Click-Through Rate (CTR) prediction in multi-field categorical datasets. Our work involves a comprehensive review of seminal models, implementation, and a Proof-of-Concept demonstration.

## Model Overview


## Requirements
python>=3.6  
pytorch>=1.10  
fuxictr==2.0.1 or lastest  
PyYAML  
pandas  
scikit-learn  
numpy  
h5py  
tqdm  



## Model Overview

This project investigates the progression of Factorization Machine-based models designed to tackle the challenge of modeling feature interactions in sparse, multi-field categorical data. We review four key contributions:

1.  **Factorization Machines (FM):** The foundational model that introduced factorized pairwise interactions, enabling generalization in sparse data.
2.  **Field-aware Factorization Machines (FFM):** Enhanced FMs by learning multiple latent vectors for each feature—one for each field it might interact with, thereby capturing field-specific interaction patterns.
3.  **Field-weighted Factorization Machines (FwFM):** Offered a more parameter-efficient approach to field-awareness by reverting to a single latent vector per feature (like FM) but introducing learnable scalar weights for each pair of fields to modulate interaction strength.
4.  **Field-matrixed Factorization Machines (FmFM/FM²):** Further generalized field-aware interactions by employing a full learnable matrix for each field pair to transform latent vectors, allowing for richer and more expressive interaction modeling. The FmFM framework also provides a unified perspective on FM and FwFM as constrained cases.

Our work includes an analysis of their architectures, mathematical underpinnings, parameter efficiency, and the evolution of how they handle field information.

## Acknowledgements
We would like to thank the authors of the seminal papers on FM, FFM, FwFM, and FmFM, whose work formed the basis of this project. We also acknowledge the FuxiCTR library for providing useful tools and implementations.


