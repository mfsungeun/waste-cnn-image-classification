# Waste Image Classification using CNN
This repository contains code for comparing and evaluating various CNN classification models on a waste image dataset.

<br />

## 🌱 Introduction
This project was completed by Agnes Song and Asmita Rokaya as part of the CSE 5717 - Big Data Analytics course in UCONN Data Science Master's program. The project applies the concepts and skills learned throughout the course, with focus on image classification.

<br />

## 📍 Objective
This project aims to develop an automated waste image classification system using the Hugging Face waste image dataset. The project uses a dataset of 3,263 images to train and test various models. We explored various deep learning models to identify the most accurate model for classifying multiple types of waste. The selection of the optimal model was based on a combination of test accuray, model complexity, and training time.

<br />

## 📶 Dataset
The dataset used for this project is the [Waste Image Dataset from Hugging Face](https://huggingface.co/datasets/rootstrap-org/waste-classifier) 🤗. It includes images categorized into seven types of waste - cardboard, compost, glass, metal, paper, plastic, trash.

   ![image](https://github.com/mfsungeun/waste-cnn-image-classification/assets/99304990/08a0276a-07c8-43d6-9586-047cd012e060)

   _Sample dataset_

<br />

## 🛠️ Methodology
1. **Data Preparation**: Setting up the dataset for model training. The preparation process includes image resizing, scaling, and data augmentation. The dataset prepration method slightly varied depending on the model employed.
2. **Model Training**:
    - CNN Base Model
    - CNN Base Model with Data Augmentation
    - Xception Model
      - Data Augmentation w/o Selective Layer Training
      - Data Augmentation w/ Selecitve Layer Training
      - Feature Extraction
    - VGG16 Model with Feature Extraction
    - ResNet50 Model with Feature Extraction
3. **Model Evaluation**: Above models were evaluated based on accuracy score, model complexity, and training time.


<br />

## 📁 Project Structure
```
.
│   README.md
│   LICENSE
│   waste_classification.ipynb
```
<br />

## 📊 Results
- Our initial models showed low accuracy.
- Data augmentation did not improve the performance of the base CNN model, possibly due to overfitting.
- Pre-trained models like VGG16, ResNet50, and Xception showed remarkable performance with feature extraction. The chart below shows comparison of all models explored.

  | Model                                          | Test Accuracy | # of Parameters | Training Time (s) |
  |------------------------------------------------|---------------|-----------------|-------------------|
  | CNN (Base Model)                               | 0.6135        | 59,079,751      | 568               |
  | CNN with Data Augmentation                     | 0.5844        | 59,079,751      | 828               |
  | Xception w/ Data Augmentation, w/o SLT         | 0.8635        | 21,914,159      | 821               |
  | Xception w/ Data Augmentation, w/ SLT          | 0.8742        | 21,914,159      | 829               |
  | Xception with Feature Extraction               | 0.9770        | 33,556,487      | 248               |
  | VGG16 with Feature Extraction                  | 0.9969        | 8,390,663       | 230               |
  | ResNet50 with Feature Extraction               | 0.9969        | 33,556,487      | 256               |

- We selected VGG16 model with feature extraction as our final model for its highest accuracy and reasonable training time.

<br />

## 💭 Future Improvements
Future improvements on this project would include:
- include more diverse waste images beyond seven categories
- address dataset imbalance by adding more images to underreprsented categories (glass)
- further optimization on model parameters
- evaluate models for real-time application

