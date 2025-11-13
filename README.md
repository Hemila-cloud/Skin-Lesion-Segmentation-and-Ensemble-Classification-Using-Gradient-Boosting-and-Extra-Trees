# Enhanced Active Contour–Based Skin Lesion Segmentation and Ensemble Classification Using Gradient Boosting and Extra Trees

### Dataset Access
Due to GitHub file size limits, the full PH2 and ISIC datasets are stored externally:
- [PH2 Dataset (Google Drive)][https://drive.google.com/drive/folders/13KYOi7hkrJKNqfIJvSmXZhbcEEEY7v8V?usp=sharing]
- [ISIC Dataset (Google Drive)][https://drive.google.com/drive/folders/14OYVWOlwl6MRsldEBUCHalPpfMJ2QNak?usp=sharing]

## Output Visuals

Due to GitHub file size limits, the full PH2 and ISIC datasets feature extraction visuals are stored externally:

PH2 train results & feature extraction
[https://drive.google.com/drive/folders/1Gg_s-bT3Zog7d1pwqOkL0rSJ38spc9S4?usp=sharing]

[https://drive.google.com/drive/folders/1xP7ggJE0K-evkJakWEm2kPZqzDpaftDC?usp=sharing]

ISIC train results & feature extraction visuals
[https://drive.google.com/drive/folders/1kE_3k3lJi1ky58OVGtwiyAyuBDtmX1Cb?usp=sharing]

[https://drive.google.com/drive/folders/1lnNIck5CwS2yylyS0-eG2q-b3KczuNeo?usp=sharing]

ISIC test results 
[https://drive.google.com/drive/folders/1kJ4DARCcqKUnSIK2WMQvehiOKz6-g-Gf?usp=sharing]


## Abstract:

Early detection of melanoma through dermoscopic analysis is vital for improving survival rates, yet manual assessment remains subjective and error-prone. This work presents an automated framework for accurate skin lesion segmentation and classification, inspired by the Kullback–Leibler Level Set (KL–LS) model. The proposed system enhances the original method by integrating **Laplacian contrast** and **distance transform** for improved boundary localization, and a **multi-scale LBP with entropy and intensity ratio** for robust feature representation. To overcome limitations of traditional classifiers such as KNN and SVM, ensemble models — **Gradient Boosting and Extra Trees** — are employed for efficient lesion characterization. Experimental evaluation on **PH2** and **ISIC** datasets demonstrates significant performance improvements in both segmentation accuracy and classification robustness, confirming the effectiveness of the proposed enhancements for automated melanoma detection.

## Base paper reference

“Active Contours Based Segmentation and Lesion Periphery Analysis for Characterization of Skin Lesions in Dermoscopy Images”
                                      — Riaz et al., IEEE Journal of Biomedical and Health 
## Ideas from base paper:
The base work by Riaz et al. introduced the Kullback–Leibler Level Set (KL–LS) model for probabilistic active contour segmentation of skin lesions. The method accurately refined lesion boundaries by comparing intensity distributions inside and outside the contour. It employed Local Binary Pattern (LBP) and center-corrected LBP features for texture representation and used K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) classifiers for lesion classification.

## Limitations in the base paper:
It relied on limited texture features.
It lacked robustness for high-dimensional data .
It did not incorporate ensemble or feature-level fusion strategies, resulting in restricted accuracy and generalization.

## Dataset Description
The PH2 dataset comprises 200 high-resolution dermoscopic images acquired under controlled illumination conditions. Each image is annotated by dermatology experts and categorized into common nevi, atypical nevi, and melanoma classes. Ground-truth lesion masks are provided, enabling objective evaluation of segmentation algorithms. The images have a consistent size of 768×560 pixels, captured with a magnification of 20×.

The ISIC dataset (International Skin Imaging Collaboration) contains a large collection of real-world dermoscopic images gathered from multiple clinical centers, representing high variability in lighting, skin tone, and lesion morphology. Each image is accompanied by a verified ground-truth segmentation mask and diagnosis label, making it suitable for both segmentation and classification benchmarking.

## Tools and libraries used:
The system was implemented in Python, utilizing libraries such as OpenCV for image preprocessing and segmentation, scikit-learn for ensemble classification (Gradient Boosting and Extra Trees), NumPy and pandas for numerical computation and data management, and matplotlib and tqdm for visualization and performance tracking.

## Steps to execute the code:

## 1. Set Up the Environment

Ensure Python 3.8 or above is installed.

Install all required dependencies using:

pip install -r requirements.txt


If a requirements.txt file is not available, install the key libraries manually:

pip install numpy opencv-python scikit-learn matplotlib pandas tqdm seaborn torch torchvision pytorch-tabnet scikit-image

## 2. Dataset Preparation

Download and organize the required datasets in the project folder as shown below:

KL_LS_Segmentation_Project/
│
├── datasets/
│   ├── PH2/
│   │   ├── Images/
│   │   └── GroundTruth/
│   └── ISIC/
│       ├── Images/
│       └── GroundTruth/



Download the datasets from the above google drive links:

PH2 Dataset (Google Drive)

ISIC Dataset (Google Drive)

Each dataset should contain both original dermoscopic images and ground-truth segmentation masks.

## 3. Preprocessing

Run preprocessing to remove hair artifacts and prepare the input images:

python src/segmentation.py


This step performs:

DullRazor hair removal

Min-RGB conversion for illumination normalization

Adaptive thresholding to generate initial lesion masks

Outputs are saved in:

outputs/<dataset_name>/preprocessed/
outputs/<dataset_name>/masks/

## 4. Active Contour Segmentation (KL–LS Refinement)

Refine lesion boundaries using the Kullback–Leibler Level Set (KL–LS) method:

python src/kl_ls_refine.py


This step iteratively adjusts contours to produce accurate segmentation of lesion regions.

## 5. Feature Extraction

Extract hybrid features for classification, including:

Local Binary Pattern (LBP)

Center-Corrected LBP

Laplacian Contrast

Distance Transform

Multi-Scale Entropy and Intensity Ratio

Run the following command:

python src/preprocess.py


Extracted feature files will be saved inside:

outputs/features_csv/

## 6. Classification

Run the classification module to train and evaluate models on the extracted features:

python src/classification_gradient_extra.py


Implemented classifiers include:

Gradient Boosting

Extra Trees

(Baseline comparison: KNN, SVM)

Results (Accuracy & ROC) will be stored in:

outputs/results_csv/

## 7. Evaluation and Visualization

Evaluate segmentation and classification performance and generate plots:

python src/evaluate.py


This computes:

Dice Coefficient, Hammoude Distance, and XOR Error for segmentation

Accuracy (%) and ROC (%) for classification

Visuals and summary results will be saved under:

outputs/visuals/


This will sequentially perform:
## Preprocessing → Segmentation → Feature Extraction → Classification → Evaluation
and generate all outputs, metrics, and graphs automatically.

## Output screenshots or result summary

![Result Visualization](output nd result summary.png)

## YouTube demo link






