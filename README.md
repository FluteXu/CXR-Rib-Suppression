# An Efficient and Robust Method for Chest X-ray Rib Suppression That Improves Pulmonary Abnormality Diagnosis

## 1. Manuscript
This work is published at https://www.mdpi.com/2075-4418/13/9/1652.

## 2. Brief Introduction
Background: Suppression of thoracic bone shadows on chest X-rays (CXRs) can improve the diagnosis of pulmonary disease. Previous approaches can be categorized as either unsupervised physical models or supervised deep learning models. Physical models can remove the entire ribcage and preserve the morphological lung details but are impractical due to the extremely long processing time. Machine learning (ML) methods are computationally efficient but are limited by the available ground truth (GT) for effective and robust training, resulting in suboptimal results. Purpose: To improve bone shadow suppression, we propose a generalizable yet efficient workflow for CXR rib suppression by combining physical and ML methods. Materials and Method: Our pipeline consists of two stages: (1) pair generation with GT bone shadows eliminated by a physical model in spatially transformed gradient fields; and (2) a fully supervised image denoising network trained on stage-one datasets for fast rib removal from incoming CXRs. For stage two, we designed a densely connected network called SADXNet, combined with a peak signal-to-noise ratio and a multi-scale structure similarity index measure as the loss function to suppress the bony structures. SADXNet organizes the spatial filters in a U shape and preserves the feature map dimension throughout the network flow. Results: Visually, SADXNet can suppress the rib edges near the lung wall/vertebra without compromising the vessel/abnormality conspicuity. Quantitively, it achieves an RMSE of ~0
 compared with the physical model generated GTs, during testing with one prediction in <1 s. Downstream tasks, including lung nodule detection as well as common lung disease classification and localization, are used to provide task-specific evaluations of our rib suppression mechanism. We observed a 3.23% and 6.62% AUC increase, as well as 203 (1273 to 1070) and 385 (3029 to 2644) absolute false positive decreases for lung nodule detection and common lung disease localization, respectively. Conclusion: Through learning from image pairs generated from the physical model, the proposed SADXNet can make a robust sub-second prediction without losing fidelity. Quantitative outcomes from downstream validation further underpin the superiority of SADXNet and the training ML-based rib suppression approaches from the physical model yielded dataset.
 
 ## 3. Data Open access.
 Paired rib suppressed CXRs are accessible at 
 
 ## Reference
 Pls cite our work as 
 
