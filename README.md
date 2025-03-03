# **Credit Card Fraud Detection**

## **Table of Contents**

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Deployment](#deployment)
8. [Visualizations](#visualizations)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)

---

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset consists of anonymized transaction records, with a binary classification label indicating whether a transaction is fraudulent (`1`) or non-fraudulent (`0`). The project involves data preprocessing, exploratory data analysis (EDA), model training using a neural network, and deployment using FastAPI and Azure ML.

## Features

- **Exploratory Data Analysis (EDA)**: Visualizing class distribution, feature relationships, and transaction amount patterns.
- **Data Preprocessing**: Feature selection, standardization, and splitting into training, validation, and test sets.
- **Neural Network Model**: A deep learning model trained using TensorFlow/Keras.
- **Performance Evaluation**: Accuracy, confusion matrix, classification report, and ROC-AUC score.
- **Deployment**: FastAPI for real-time predictions and Azure ML for cloud-based deployment.

## Installation

To run this project, ensure you have the required dependencies installed:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn fastapi uvicorn azureml-core
```

## Dataset

The dataset used in this project is `creditcard_2023.csv`, which contains anonymized transaction details. Key steps performed on the dataset include:

1. Dropping unnecessary columns (`id`, `Amount`)
2. Standardizing feature values
3. Splitting into train, validation, and test sets

Dataset url: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/download?datasetVersionNumber=1

## Model Training

The neural network model consists of:

- 3 hidden layers with **ReLU** activation
- **Dropout** layers for regularization
- A **sigmoid** output layer for binary classification
- **Adam** optimizer and **binary\_crossentropy** loss function

### Training Command

```python
history = model.fit(X_train_scaled, y_train, epochs=6, batch_size=32, validation_data=(X_val_scaled, y_val))
```

## Evaluation

After training, the model is evaluated using:

```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob))
```

## Deployment

### FastAPI

To deploy the model as an API using **FastAPI**:

1. Create a FastAPI app (`app.py`)
2. Define an endpoint for fraud detection
3. Run the server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

### Azure ML

For cloud-based deployment:

1. Register the model in **Azure ML**
2. Deploy as a web service using an inference configuration

## Visualizations

Key visualizations include:

- **Class Distribution**: Distribution of fraudulent vs. non-fraudulent transactions.
- **Transaction Amount Distribution**: Histograms and box plots.
- **Correlation Heatmap**: Identifying highly correlated features.

## Contributing

If you’d like to contribute, feel free to fork this repository and submit a pull request.

## License

This project is licensed under the **MIT License**.

## Contact

For any questions, feel free to reach out!

