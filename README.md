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

## 🌐 Live Demo

🔗 **Try the Application Here**

https://mobilepredictionproject-xxfdne9wke5hzsjeztaw5k.streamlit.app/

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

## ⚙️ How The System Works

```text
User Inputs Mobile Specifications
                │
                ▼
       Data Preprocessing
                │
                ▼
       Trained ML Model
                │
                ▼
      Price Category Prediction
                │
                ▼
          Final Result

# In detail this project

# Why This Project?

The smartphone industry offers thousands of devices with different specifications and price points. Determining the price category of a mobile phone based on its technical features can help manufacturers, retailers, and consumers better understand market positioning.

This project was developed to:

* Understand real-world machine learning workflows
* Practice classification problems
* Analyze relationships between smartphone features and pricing
* Improve data preprocessing skills
* Perform exploratory data analysis
* Build predictive machine learning models
* Create a practical portfolio project

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

# Project Objectives

The main objectives of this project are:

* Predict mobile phone price category
* Analyze feature importance
* Understand factors affecting smartphone pricing
* Compare machine learning algorithms
* Improve classification model performance
* Generate insights from mobile specifications

---

# How the System Works

## Step 1: Data Collection

The dataset containing mobile specifications is loaded into Python using Pandas.

---

## Step 2: Data Preprocessing

The dataset is cleaned and prepared for machine learning.

Tasks include:

* Checking missing values
* Removing duplicates
* Data validation
* Feature selection

---

## Step 3: Exploratory Data Analysis (EDA)

The dataset is analyzed to understand:

* Distribution of mobile specifications
* Relationship between RAM and price
* Battery power trends
* Screen resolution effects
* Connectivity feature impact

---

## Step 4: Feature Engineering

Relevant features are selected and prepared for model training.

Examples:

* RAM
* Battery Power
* Internal Memory
* Processor Cores
* Camera Quality

---

## Step 5: Model Training

Machine learning algorithms learn patterns between smartphone specifications and price categories.

The model identifies how different hardware features influence the final price range.

---

## Step 6: Prediction

The user enters mobile specifications.

Example:

Battery Power: 2000

RAM: 4096

Internal Memory: 64

Processor Cores: 8

Primary Camera: 48 MP

The trained model predicts:

Price Range = High Cost

---

## Step 7: Result Generation

The system returns the predicted mobile price category based on the provided specifications.

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

# Applications

This project can be used for:

* Smartphone Price Prediction
* Market Research
* Product Categorization
* E-Commerce Analysis
* Consumer Decision Support
* Educational Machine Learning Projects

---

# Learning Outcomes

Through this project, the following skills were developed:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Data Visualization
* Feature Engineering
* Classification Models
* Model Evaluation
* Machine Learning Workflow
* Python Development

---

# Future Improvements

Possible future enhancements:

* Streamlit Web Application
* Mobile Recommendation System
* Deep Learning Models
* Hyperparameter Tuning
* Real-Time Price Prediction API
* Feature Importance Dashboard
* Deployment on Cloud Platforms

---

# Conclusion

The Mobile Price Range Prediction System demonstrates how machine learning can classify smartphones into different price categories using hardware and technical specifications. By analyzing key features such as RAM, battery power, processor performance, memory, and display quality, the system provides accurate price range predictions and valuable insights into smartphone pricing patterns.
