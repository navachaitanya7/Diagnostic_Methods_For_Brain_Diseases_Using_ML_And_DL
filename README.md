# Diagnostic_Methods_For_Brain_Diseases_Using_ML_And_DL

# 🧠 Brain Disease Detection from MRI Scans using Machine Learning

## 📌 Introduction

This project aims to develop an intelligent system that assists in the **early detection** of brain diseases using **MRI scans**. It combines **deep learning** and **classical machine learning** techniques to analyze complex brain images and predict conditions such as:

- Alzheimer's Disease  
- Brain Tumors  
- Epilepsy  
- Parkinson's Disease  
- Normal (Healthy Brain)

---

## 🎯 Purpose and Motivation

Diagnosing neurological diseases can be challenging due to complexity and image interpretation inconsistencies. This system is designed to:

- ✅ **Improve Accuracy** – Detect subtle features in MRI scans beyond human visibility.  
- ⏱️ **Enable Early Detection** – Speed up the diagnosis process for better treatment outcomes.  
- 👨‍⚕️ **Support Clinicians** – Provide data-driven insights to aid clinical decisions.

---

## 🔧 System Workflow

1. **Upload Image** – User uploads an MRI scan via a web interface.  
2. **Feature Extraction** – An EfficientNetB0 CNN extracts key features from the image.  
3. **Classification** – Classical ML algorithms predict the disease class.  
4. **Display Result** – The final diagnosis is displayed to the user.

---

## ⚙️ Technologies Used

### 🧠 Algorithms

- **EfficientNetB0 (CNN):** Deep learning model for image feature extraction.  
- **Classical ML Models:**  
  - Random Forest  
  - Support Vector Machine (SVM)  
  - Gaussian Naive Bayes  
  - Logistic Regression  

### 💻 Libraries

- `NumPy` – Numerical computations  
- `scikit-learn` – ML algorithms and evaluation  
- `TensorFlow/Keras` – Deep learning framework  
- `Pillow (PIL)` – Image preprocessing  
- `Streamlit` – Web interface  
- `Matplotlib` – Data visualization

---

## 🧩 Project Structure

```
📁 BrainDiseaseDetection/
│
├── app.py               # Streamlit frontend
├── load_images.py       # Image loading & preprocessing
├── model.py             # Feature extraction & model training
├── predict.py           # Prediction logic
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.7 or higher  
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone <repository_url>
cd <repository_directory>

# Create and activate virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

1. Run the web app:
   ```bash
   streamlit run app.py
   ```
2. Open the local URL in your browser.  
3. Upload an MRI image.  
4. View the predicted brain disease diagnosis.

---

## 🗂️ Dataset

Ensure your dataset is structured like this:

```
📁 Dataset/
│
├── Alzheimer/
├── BrainTumor/
├── Epilepsy/
├── Parkinson/
└── Normal/
```

Update the path in `load_images.py` to match your local dataset directory.

---

## ⚠️ Limitations & Future Enhancements

- 📈 Improve dataset quality and size for better accuracy.  
- 🤝 Integrate Explainable AI for transparency in model predictions.  
- 🧪 Add real-time processing for faster diagnosis.  
- 🛡️ Address data privacy and ethical concerns before clinical deployment.  
- 🧬 Explore hybrid and ensemble modeling approaches.

---

## 👥 Contributors

- V. Karthikeya (20B81A05I6)  
- Y. LalithNivas (20B81A05I7)  
- Y. Nava Chaitanya (20B81A05I8)  
- Y. HemaSri (20B81A05I9)  
- Y. Harshitha (20B81A05J0)

---

## 🧑‍🏫 Supervisor

**G. Monika Devi, M.Tech**  
Assistant Professor, CSE  
Sir C.R. Reddy College of Engineering

---

Let me know if you'd like this formatted into a markdown file (`README.md`) ready for GitHub!
