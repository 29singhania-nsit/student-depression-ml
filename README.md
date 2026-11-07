# Student Depression Classification

## A. Problem Statement

The objective of this project is to build and compare different machine learning classification models for predicting student depression based on demographic, academic, social, health and living-related attributes.

## B. Dataset Description

**Dataset:** Student Depression Dataset  
**Problem Type:** Binary Classification  
**Target Variable:** Depression  
**Target Classes:** Yes and No  
**Number of Input Features:** 12

The dataset contains student-related demographic, academic, social, health and lifestyle information. The target variable indicates whether a student is classified as having depression or not.

## C. GitHub Repository

**GitHub Link:** ADD GITHUB LINK HERE

## D. Models Used

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## E. Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9709 | 0.9931 | 0.9612 | 0.9292 | 0.9449 | 0.9254 |
| Decision Tree | 0.9966 | 0.9938 | 1.0000 | 0.9875 | 0.9937 | 0.9915 |
| KNN | 0.9474 | 0.9786 | 0.9289 | 0.8708 | 0.8989 | 0.8642 |
| Naive Bayes | 0.9496 | 0.9875 | 0.8884 | 0.9292 | 0.9084 | 0.8740 |
| SVM | 0.9877 | 0.9965 | 0.9872 | 0.9667 | 0.9768 | 0.9686 |

## F. Model-wise Observations

### Logistic Regression

Logistic Regression performed very well with an accuracy of 97.09% and an AUC of 0.9931. The F1 score of 0.9449 also indicates strong classification performance.

### Decision Tree

Decision Tree achieved the best overall performance. It obtained an accuracy of 99.66%, precision of 100%, recall of 98.75%, F1 score of 0.9937 and MCC of 0.9915.

### K-Nearest Neighbors

KNN achieved an accuracy of 94.74%. Its recall and F1 score were lower than the other high-performing models. Therefore, KNN showed the lowest overall performance among the five models in this experiment.

### Naive Bayes

Naive Bayes achieved an accuracy of 94.96% and an AUC of 0.9875. Its recall was 92.92%, while its precision was comparatively lower at 88.84%.

### Support Vector Machine

SVM achieved excellent performance with an accuracy of 98.77% and an F1 score of 0.9768. SVM achieved the highest AUC among all five models at 0.9965.

## G. Overall Observation

Decision Tree achieved the best overall test-set performance based on Accuracy, Precision, Recall, F1 Score and MCC. SVM achieved the highest AUC value of 0.9965. Therefore, Decision Tree was the best overall performing model for this experiment, while SVM also demonstrated excellent classification performance.
