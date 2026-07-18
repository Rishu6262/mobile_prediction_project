# 📱 Mobile Price Range Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Visualization-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

### 🚀 Machine Learning-Based Smartphone Price Category Prediction

*A complete end-to-end Machine Learning project that predicts the price category of smartphones using their hardware specifications and technical features.*

</div>

---

## 🌐 Live Demo

🔗 **Try the Application Here**

https://mobilepredictionproject-xxfdne9wke5hzsjeztaw5k.streamlit.app/

---

# 📌 Project Overview

The **Mobile Price Range Prediction System** is an end-to-end **Machine Learning Classification** project developed using **Python**, **Scikit-Learn**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **Streamlit**. The objective of this project is to accurately predict the **price category** of a smartphone based on its hardware specifications and technical characteristics.

The application analyzes a wide range of smartphone features—including **battery capacity, RAM, internal storage, processor cores, clock speed, camera quality, screen dimensions, display resolution, connectivity options, and other hardware specifications**—to determine the most appropriate price range. By learning patterns from historical mobile device data, the trained machine learning model can classify smartphones into predefined pricing categories with high accuracy.

This project demonstrates the complete **Machine Learning lifecycle**, beginning with data collection and preprocessing, followed by exploratory data analysis (EDA), feature engineering, model training, performance evaluation, and deployment through an interactive **Streamlit web application**. It highlights how machine learning can assist in solving real-world business problems related to product pricing and market segmentation.

The prediction system evaluates multiple smartphone attributes, including:

* 🔋 Battery Power
* 🧠 RAM Capacity
* 💾 Internal Storage
* ⚙️ Processor Cores & Clock Speed
* 📷 Front & Rear Camera Quality
* 📱 Screen Size & Display Resolution
* 🌐 Connectivity Features (Wi-Fi, Bluetooth, 3G, 4G)
* 📞 Talk Time & Battery Efficiency
* 📲 Touch Screen Support
* 📦 Device Weight & Mobile Depth

Based on these specifications, the trained classification model predicts one of the following smartphone price categories:

| Price Range | Category                |
| ----------- | ----------------------- |
| **0**       | 💰 Low Cost             |
| **1**       | 💰💰 Medium Cost        |
| **2**       | 💰💰💰 High Cost        |
| **3**       | 💰💰💰💰 Very High Cost |

This project serves as an excellent example of applying **Machine Learning**, **Data Analysis**, and **Predictive Analytics** to a real-world classification problem. It demonstrates practical skills in data preprocessing, feature engineering, model development, evaluation, deployment, and user interface design, making it a valuable portfolio project for aspiring **Data Scientists**, **Machine Learning Engineers**, and **Python Developers**.

---

## 🎯 Problem Statement

Smartphones come with a wide range of hardware specifications, making it difficult to estimate their price category accurately. Factors such as **RAM, battery power, processor, storage, camera quality, and display resolution** all influence a device's market value.

This project aims to develop a **Machine Learning classification model** that predicts the price category of a smartphone based on its technical features. The system provides fast and accurate predictions, helping users understand how different specifications affect smartphone pricing.


---

## 📊 Dataset Information

| Property | Value |
|-----------|--------|
| Dataset Name | Mobile Price Range Dataset |
| Records | 2000 |
| Features | 21 |
| Problem Type | Classification |
| Target Variable | price_range |

---

## ⚙️ How the System Works

```text
User Enters Mobile Specifications
              │
              ▼
      Data Preprocessing
              │
              ▼
    Trained Machine Learning Model
              │
              ▼
     Price Category Prediction
              │
              ▼
         Display Final Result
```

---

## 💡 Why This Project?

With thousands of smartphones available in the market, estimating a device's price category based on its specifications can be challenging. This project uses **Machine Learning** to analyze smartphone features and accurately predict the appropriate price range, making the process faster and more reliable.

### 🎯 Project Objectives

* Build a Machine Learning classification model
* Predict smartphone price categories accurately
* Analyze the impact of hardware specifications on pricing
* Perform data preprocessing and exploratory data analysis (EDA)
* Compare classification algorithms and evaluate their performance
* Develop a practical end-to-end Machine Learning portfolio project


---

# Dataset Features

### Hardware Specifications

* battery_power
* ram
* n_cores
* clock_speed
* int_memory
* mobile_wt

### Camera Features

* fc (Front Camera)
* pc (Primary Camera)

### Display Features

* px_height
* px_width
* sc_h
* sc_w

### Connectivity Features

* bluetooth
* dual_sim
* three_g
* four_g
* wifi

### Additional Features

* talk_time
* touch_screen
* m_dep

### Target Feature

* price_range

---

## 🎯 Project Objectives

The primary objectives of this project are to:

* 📱 Predict the price category of a smartphone based on its technical specifications.
* 🤖 Develop an accurate Machine Learning classification model.
* 📊 Analyze the relationship between mobile features and pricing.
* 🔍 Identify the most important features influencing smartphone prices.
* 🧹 Perform data preprocessing and exploratory data analysis (EDA).
* ⚡ Compare different Machine Learning algorithms to achieve the best performance.
* 📈 Evaluate the model using classification metrics such as Accuracy, Precision, Recall, and F1-Score.
* 🌐 Deploy the trained model as an interactive Streamlit web application.
* 💼 Build a real-world portfolio project demonstrating end-to-end Machine Learning skills.

---

## ⚙️ How the System Works

The Mobile Price Range Prediction System follows a complete Machine Learning pipeline, from data preparation to price prediction.

### **Step 1: Data Collection**

The mobile price dataset containing various smartphone specifications is loaded into Python using **Pandas** for further analysis and processing.

---

### **Step 2: Data Preprocessing**

The dataset is cleaned and prepared to ensure high-quality input for the machine learning model.

**Tasks Performed:**

* Check for missing values
* Remove duplicate records
* Validate dataset integrity
* Select relevant features

---

### **Step 3: Exploratory Data Analysis (EDA)**

The dataset is explored to identify patterns, trends, and relationships between smartphone features and their price categories.

**Analysis Includes:**

* Distribution of smartphone specifications
* RAM vs. Price Range
* Battery Power Analysis
* Camera Feature Analysis
* Display Resolution Analysis
* Connectivity Feature Analysis

---

### **Step 4: Model Training**

Multiple Machine Learning classification algorithms are trained using the processed dataset to learn the relationship between smartphone specifications and their corresponding price categories.

---

### **Step 5: Price Prediction**

Users enter the smartphone specifications through the Streamlit web application. The trained model processes the input data and predicts the appropriate price category.

**Example Input:**

* 🔋 Battery Power: **2000 mAh**
* 🧠 RAM: **4096 MB**
* 💾 Internal Memory: **64 GB**
* ⚙️ Processor Cores: **8**
* 📷 Primary Camera: **48 MP**

**Predicted Output:**

* 💰 **High Cost**

---

### **Step 6: Result Generation**

Finally, the system displays the predicted smartphone price category, enabling users to estimate the market value based on the provided hardware specifications.


---

# Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

# Machine Learning Algorithms

Possible models used:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* XGBoost

---

# Python Concepts Used

This project demonstrates:

* Data Cleaning
* Exploratory Data Analysis
* Data Visualization
* Classification Algorithms
* Feature Engineering
* Model Training
* Model Evaluation
* Prediction Systems
* Python Programming

---

# Project Structure

```bash
Mobile_Price_Prediction/
│
├── mobile.csv
├── main.ipynb
├── app.py
├── model.pkl
├── scaler.pkl
├── outputs/
└── README.md
```

---

# Analysis Performed

### RAM vs Price Range

Analyze how RAM affects smartphone pricing.

---

### Battery Power Analysis

Study battery capacity across price categories.

---

### Camera Analysis

Compare camera specifications among mobile categories.

---

### Screen Resolution Analysis

Evaluate the impact of display quality on price.

---

### Connectivity Analysis

Study the role of 3G, 4G, WiFi, and Bluetooth features.

---

# Model Evaluation

Evaluation metrics may include:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

These metrics help measure prediction performance.

---

# 🌍 Applications

The **Mobile Price Range Prediction System** can be applied in various real-world scenarios, including:

* 📱 Smartphone Price Prediction
* 🛒 E-Commerce Product Categorization
* 📊 Market Research and Pricing Analysis
* 🏪 Retail Inventory Management
* 💡 Consumer Decision Support
* 📈 Business Intelligence and Market Insights
* 🎓 Educational Machine Learning Projects
* 💼 Machine Learning Portfolio Demonstrations

---

# 📚 Learning Outcomes

This project provided practical experience in building an end-to-end Machine Learning application. Through its development, the following skills were strengthened:

* 🧹 Data Cleaning and Preprocessing
* 📊 Exploratory Data Analysis (EDA)
* 📈 Data Visualization
* ⚙️ Feature Engineering
* 🤖 Machine Learning Classification
* 📉 Model Evaluation and Performance Analysis
* 🔍 Feature Importance Analysis
* 🌐 Streamlit Web Application Development
* 🐍 Python Programming and Scikit-Learn
* 🚀 End-to-End Machine Learning Workflow

---

# 🚀 Future Improvements

The project can be further enhanced by adding advanced features such as:

* 🌐 Real-Time Price Prediction API
* 📱 Mobile Recommendation System
* 🧠 Deep Learning-Based Classification Models
* ⚙️ Hyperparameter Tuning for Better Accuracy
* 📊 Interactive Feature Importance Dashboard
* ☁️ Cloud Deployment (AWS, Azure, or GCP)
* 📈 Advanced Analytics and Visualization
* 🔄 Automatic Model Retraining with New Data
* 🌍 Support for Live Smartphone Market Data

---

# 📌 Conclusion

The **Mobile Price Range Prediction System** is an end-to-end Machine Learning project that accurately classifies smartphones into different price categories using their hardware and technical specifications. By leveraging data preprocessing, exploratory data analysis, feature engineering, and classification algorithms, the system delivers reliable price predictions while identifying the key factors that influence smartphone pricing.

This project demonstrates practical skills in **Python**, **Machine Learning**, **Scikit-Learn**, **Data Analysis**, and **Streamlit deployment**, making it a valuable portfolio project for aspiring **Data Scientists**, **Machine Learning Engineers**, and **Python Developers**.
