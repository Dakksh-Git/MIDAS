Based on what MIDAS actually builds upon, these are the 5 most relevant — not just cited, but directly foundational to the system:

---

*[14] Sundararajan, Taly & Yan (2017)*
"Axiomatic Attribution for Deep Networks" — ICML 2017
The theoretical foundation for the entire explainability component. IG is not just cited — it is the attribution method implemented in src/gradcam.py. The two axioms (sensitivity and implementation invariance) are what justify choosing IG over Grad-CAM.

---

*[5] He, Zhang, Ren & Sun (2016)*
"Deep Residual Learning for Image Recognition" — IEEE CVPR 2016
The backbone of every branch in MIDAS is a 3D adaptation of ResNet-18. Without this paper, there is no MIDAS architecture. The residual connection is what makes training 18 layers on a modest dataset feasible.

---

*[10] Dolz et al. (2019)*
"HyperDense-Net: A Hyper-Densely Connected CNN for Multi-Modal Image Segmentation" — IEEE TMI 2019
Directly justifies the parallel-branch intermediate fusion strategy over early fusion. The core architectural decision in MIDAS — four independent encoders before concatenation — is grounded in the evidence this paper provides.

---

*[16] Menze et al. (2015)*
"The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)" — IEEE TMI 2015
The BraTS2020 dataset, derived from this benchmark series, provides both the Malignant and Benign classes — the two largest and most clinically critical categories in MIDAS. The preprocessing conventions (co-registered sequences, skull-stripped volumes, HGG/LGG labels) adopted in MIDAS follow BraTS standards.

---

*[22] Morin et al. (2023)*
"LUMIERE Dataset: Longitudinal Glioblastoma MRI with Expert RANO Evaluation" — Scientific Data 2023
Without LUMIERE, the Scar class would have been represented by two patients and augmented copies — making the 98.90% Scar accuracy meaningless. This paper is the reason the Scar class has genuine clinical validity in the system.