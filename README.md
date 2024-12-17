# DL-for-Detection-of-Coronary-Artery-Stenosis

Project Overview
This repository contains the code for preprocessing and training a deep learning model on curved multiplanar reformations (CMR) images. The project aims to enhance diagnostic accuracy for coronary artery disease by leveraging routine clinical imaging data.

Key Features
Preprocessing: The preprocessing pipeline cleans and prepares CMR images to optimize them for deep learning models. This includes normalization, artifact removal, and other image enhancements necessary for effective model training.

Data Augmentation: Techniques such as rotation, flipping, and random cropping are used to increase dataset diversity and improve model generalization.
Model Training: Code for training deep learning models (e.g., EfficientNet) on the processed CMR images to detect coronary artery stenoses. The model training strategy focuses on handling real-world clinical data and achieving high diagnostic accuracy.

Instructions for Use

Setup:
Clone this repository to your local machine.
Ensure Python and the required libraries (e.g., TensorFlow, NumPy, etc.) are installed.
Set up a virtual environment to manage dependencies.

Data Preparation:
Place your CMR image data in the appropriate directory.
The dataset should be organized into appropriate folders (e.g., by diagnosis).
Preprocessing steps (normalization, augmentation) will be automatically applied during training.

Training the Model:
Modify the configuration file to specify the training parameters such as batch size, learning rate, number of epochs, etc.
Run the training script (train_model.py) to start the training process.
The model will be trained using the augmented and preprocessed data to detect coronary artery stenoses.

Model Evaluation:
Evaluate the trained model on validation data using metrics such as accuracy, area under the curve (AUROC), and other performance metrics.
The evaluation results will provide insights into the model’s effectiveness on both normal and anomalous cases.

Future Work:
Expand the dataset to include more cases and diverse patient populations.
Explore additional preprocessing techniques and advanced augmentation strategies.
Implement external validation on independent datasets to assess the model’s robustness.
Address challenges such as handling anomalous vessel origins and improve interpretability of model decisions through heatmaps and feature visualization techniques.

Acknowledgements
This project was inspired by the need for improved diagnostic tools in routine clinical workflows.
Special thanks to the radiologists who provided clinical images for this study and contributed to dataset annotations.

Notes
Due to privacy and security concerns, the data used in this study cannot be shared. However, the preprocessing code and model training strategies are made available to facilitate reproducibility and enable independent validation by other research groups.
