# Study on Adversarial Attacks on COVID-19 CT Scan Classification Neural Network

This study explores the impact of five different adversarial attacks on a neural network classifier trained to distinguish between COVID-19 and non-COVID-19 lung CT scans. The attacks employed in this study are FGSM (used for poisoning), BadNet, Clean Label, WaNet, and Hidden Trigger Backdoor.

The neural network model under investigation has been trained on a dataset comprising CT scan images of lungs, with labels indicating whether the scan is COVID-19 positive or negative. The goal of the study is to assess the robustness of the model against adversarial attacks and to evaluate the effectiveness of various attack strategies in compromising the model's performance.

To use this code you have to download the dataset https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset and structure the code directory as follow:
.
└── Datasets/

    ├── Train/
    
    └── Test/

## Dataset & Techniques Used

1. Dataset Preparation: The dataset used in this study consists of a collection of CT scan images, annotated with COVID-19 infection status, you can found it to this link https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset.

2. Adversarial Attacks: Five different adversarial attacks were applied to the trained neural network model:
   - FGSM (Fast Gradient Sign Method)
   - BadNet
   - Clean Label
   - WaNet
   - Hidden Trigger Backdoor
