# 🔐 CAPTCHA Recognition using Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-NeuralNetwork-red?logo=keras)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue?logo=docker)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?logo=amazonaws)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-black?logo=githubactions)

A deep learning based system that automatically recognizes and solves CAPTCHA images using a **Convolutional Neural Network (CNN)**.  
The trained model is integrated into a **demo web application** that captures CAPTCHA images and predicts the text in real time.

The project demonstrates the **complete ML lifecycle** including data preprocessing, model training, model deployment using Docker, and automated **CI/CD pipelines using GitHub Actions on AWS**.

---

# 🚀 Project Overview

CAPTCHAs are widely used to differentiate between human users and automated bots. This project focuses on building a **deep learning model capable of recognizing CAPTCHA text automatically**.

### The system:

- Processes CAPTCHA image datasets
- Trains a CNN model to recognize characters
- Exports the trained model
- Integrates the model into a web application
- Deploys the application on AWS using Docker
- Automates deployment using CI/CD pipelines

---

# 🧠 Model Architecture

The model is built using a **Convolutional Neural Network (CNN)** designed for image recognition tasks.

### Key components used in the architecture:

- Convolutional Layers for feature extraction
- MaxPooling Layers for dimensionality reduction
- Multiple hidden layers to learn complex patterns
- Techniques to control overfitting during training
- Fully Connected Layers for final prediction

The model learns to extract visual patterns from CAPTCHA images and convert them into predicted characters.

---

# 🏗 System Architecture

```
        +----------------------+
        |   CAPTCHA Dataset    |
        |      (Kaggle)        |
        +----------+-----------+
                   |
                   v
        +----------------------+
        |   Data Preprocessing |
        |  Cleaning, ETL, etc  |
        +----------+-----------+
                   |
                   v
        +----------------------+
        |   CNN Model Training |
        | TensorFlow / Keras   |
        +----------+-----------+
                   |
                   v
        +----------------------+
        |   Export Model (.h5) |
        +----------+-----------+
                   |
                   v
        +----------------------+
        |   Demo Web App       |
        | CAPTCHA Prediction   |
        +----------+-----------+
                   |
                   v
        +----------------------+
        | Docker Container     |
        +----------+-----------+
                   |
                   v
        +----------------------+
        | AWS ECR (Image Repo) |
        +----------+-----------+
                   |
                   v
        +----------------------+
        | AWS EC2 Deployment   |
        +----------------------+

```

---

# 📊 Dataset

The dataset used for training the model was sourced from **Kaggle CAPTCHA datasets**.

## Dataset Processing

Before training, the dataset went through a preprocessing pipeline:

- Image normalization
- Label extraction from image names
- Dataset cleaning
- Train-test splitting
- Feature preparation for CNN input

The processed dataset was then used to train and evaluate the model.

---

# ⚙️ Machine Learning Pipeline

The project follows a structured ML pipeline:

## 1️⃣ Data Preprocessing

- Dataset cleaning
- Image resizing
- Label extraction
- Train-test split

## 2️⃣ Model Training

- CNN based architecture
- Hyperparameter tuning
- Overfitting control

## 3️⃣ Model Export

After training, the model is exported as:

```
model.h5
```

This allows the trained model to be reused for inference.

---

# 🌐 Web Application (Demo)

A simple demo web application was created to demonstrate the model.

## Workflow

1. CAPTCHA image appears on the web interface
2. The image is captured by the backend
3. The trained CNN model processes the image
4. The model predicts the CAPTCHA text
5. The predicted output is displayed

This demonstrates how the trained deep learning model can be used in **real-world CAPTCHA recognition scenarios**.

---

# ☁️ Deployment Architecture

The entire application was deployed on **AWS** using containerization.

## Deployment Stack

- Docker for containerizing the application  
- AWS Elastic Container Registry (ECR) for storing Docker images  
- AWS EC2 instance for running the containerized application  

## Deployment Flow

1. Application packaged using **Docker**
2. Docker image pushed to **AWS ECR**
3. Image pulled and run on **AWS EC2**

---

# 🔄 CI/CD Pipeline

A **CI/CD pipeline** was implemented using **GitHub Actions** to automate deployment.

## CI/CD Workflow

1. Code pushed to GitHub repository
2. GitHub Actions pipeline triggers automatically
3. Docker image is built
4. Image pushed to AWS ECR
5. EC2 instance pulls the latest image
6. Application is updated automatically

This ensures **continuous integration and continuous deployment**.

---

# 📈 Model Performance

### Model Accuracy

**91% Accuracy on CAPTCHA recognition**

The model was trained to effectively recognize distorted CAPTCHA characters and generalize well on unseen data.

---

# 🛠 Tech Stack

## Programming
- Python

## Machine Learning
- TensorFlow
- Keras
- CNN (Convolutional Neural Networks)

## Data Processing
- NumPy
- Pandas
- OpenCV

## Deployment
- Docker
- AWS EC2
- AWS ECR

## DevOps
- GitHub Actions
- CI/CD Pipelines

---

# 📌 Key Features

- Deep Learning based CAPTCHA recognition
- End-to-end ML pipeline implementation
- Model deployment using Docker containers
- Cloud deployment on AWS
- Automated CI/CD pipeline using GitHub Actions

---

# 👨‍💻 Author

**Samarpit Kandhari**

Computer Science Engineer | Machine Learning | Data Science | AI
