# 📱 Mobile Price Range Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

### Predict Mobile Price Categories Using Machine Learning

</div>

---

## 🌐 Live Demo

🔗 **Try the Application Here**

https://mobilepredictionproject-xxfdne9wke5hzsjeztaw5k.streamlit.app/

---

## 🚀 Project Overview

The Mobile Price Range Prediction System is a Machine Learning-based web application that predicts the price category of a smartphone using its technical specifications.

The system analyzes hardware and software-related features such as:

- 🔋 Battery Power
- 🧠 RAM
- 💾 Internal Memory
- ⚙️ Processor Cores
- 📷 Camera Quality
- 📱 Screen Resolution
- 🌐 Connectivity Features

and classifies the smartphone into one of the following categories:

| Price Range | Category |
|------------|----------|
| 0 | 💰 Low Cost |
| 1 | 💰💰 Medium Cost |
| 2 | 💰💰💰 High Cost |
| 3 | 💰💰💰💰 Very High Cost |

---

## 🎯 Problem Statement

Mobile phones come with hundreds of different specifications, making it difficult to estimate their market value accurately.

This project aims to build a machine learning model capable of predicting the price category of a smartphone based on its technical features.

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
