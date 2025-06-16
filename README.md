# Breast Cancer Prediction Using Neural Networks

This project demonstrates how deep learning can be applied to healthcare problems, specifically for the classification of breast cancer tumors as **benign** or **malignant** using diagnostic features. A neural network model is built and trained to assist in early cancer detection, a task where accuracy can literally save lives.

---

## 📦 Step 1: Loading the Dependencies
Essential Python libraries are imported to support data manipulation, modeling, and prediction:
- `pandas`, `numpy`
- `scikit-learn`
- `tensorflow.keras`
- `matplotlib`, `seaborn`

---

## 📂 Step 2: Loading the Dataset
The dataset contains diagnostic measurements for breast cancer, with labels indicating tumor type:
- `Label 1 → Benign`
- `Label 0 → Malignant`

The data is clean and well-structured, allowing us to move directly into modeling.

---

## 🧼 Step 3: Data Preprocessing
Preprocessing steps include:
- Encoding categorical labels
- Feature scaling
- Splitting the data into training and testing sets

---

## 🏗️ Step 4: Building a Neural Network
A deep learning model is constructed using `Keras`, with:
- A **Flatten layer** to reshape input features into a 1D vector
- Multiple fully connected (`Dense`) layers
- Activation functions like **ReLU** and **Sigmoid**
- **Binary cross-entropy** as the loss function
- Evaluation on **accuracy** and **loss** metrics

The model is trained to recognize complex patterns in the diagnostic features.

---

## 📊 Step 5: Visualizing Model Performance

To evaluate how well the neural network learned from the training data, **accuracy and loss curves** were plotted over 20 training epochs:

- **Accuracy Curve**:  
  Training accuracy steadily improved, while validation accuracy reached ~98% early and remained stable, indicating excellent generalization.

- **Loss Curve**:  
  Both training and validation loss decreased consistently, suggesting the model learned efficiently without overfitting.

These visualizations helped confirm the model’s robustness and effectiveness.

---

## 🔮 Step 6: Predictions
After validation, the model was used to make predictions on new input data, simulating real-world scenarios where timely classification is crucial.

---

## 📈 Step 7: Building a Predictive System
The trained model was wrapped into a simple system capable of accepting user input and providing an immediate prediction. This serves as a proof of concept for deploying AI in medical diagnostics.

---

## 🧠 Insight
Even with a relatively simple neural network architecture, the model achieved high accuracy due to the strong signal in the dataset. This reinforces the value of clean, well-structured data — and highlights how machine learning can enhance diagnostic processes when paired with domain knowledge.

---

## 📁 File Structure
- `DL1_Breast_Cancer_Predictions_with_Neural_Networks.ipynb` – Jupyter notebook containing the full workflow
- `README.md` – Documentation (this file)

---

## 🚀 Requirements
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
