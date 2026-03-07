# 🎥 YouTube Sentiment Insights

An **End-to-End MLOps Project** that analyzes YouTube comments and predicts their sentiment using machine learning models.  
This project demonstrates how to build a **production-ready machine learning pipeline** including experiment tracking, data pipelines, API deployment, Docker containerization, and CI/CD deployment on AWS.

---

# 🚀 Project Overview

YouTube videos often receive thousands of comments, making it difficult for creators to manually analyze viewer sentiment.

This project builds an automated system that:

- Collects YouTube comments
- Cleans and preprocesses text data
- Extracts features using NLP techniques
- Trains multiple machine learning models
- Tracks experiments using MLflow
- Builds reproducible ML pipelines using DVC
- Deploys the trained model via Flask API
- Integrates the model into a Chrome Extension
- Uses Docker and CI/CD for deployment

The goal is to simulate a **real-world production ML system**.

---

# 🧠 Machine Learning Approach

The project applies multiple NLP techniques to transform text data into numerical features.

## Text Vectorization

### Bag of Words (BOW)

Represents text as word frequency vectors.

Example:

### Sentence:
I love this video

### Vector representation:
[0,1,0,1,2]


---

### TF-IDF

TF-IDF improves Bag of Words by giving higher importance to rare words and reducing the importance of common words.

---

# 🤖 Models Used

Multiple machine learning models were trained and evaluated.

Examples include:

- Logistic Regression
- Naive Bayes
- Linear Models

---

# 🔗 Ensemble Learning

To improve performance, the project implements a **Stacking Classifier**.

Stacking combines predictions from multiple models and trains a meta-model on top of them to produce better predictions.

---

# ⚙️ MLOps Pipeline

The machine learning pipeline is automated using **DVC (Data Version Control)**.

Pipeline stages:
