# Predicting Heart Disease Risk using Python and Machine Learning

## Background

The Management of Zend Hospital recently observed a spike in hospital admissions due to chronic heart disease, which has resulted in increased operational costs and a significant impact on patients’ health.

## Goals
+ Conduct exploratory data analysis to uncover trends and patterns
+ Identify factors influencing heart disease incidence using historical data and specific features
+ Perform feature importance analysis
+ Build a robust machine-learning model using historical data to predict heart disease risk

## Methodology

### Data Source

Patient data was extracted from an Electronic Health Records System

### 1. Data Preprocessing for Exploratory Data Analysis

+ Loaded data
+ Performed preliminary exploration to view features and the number of records
+ Checked and replaced missing values
+ Checked for duplicates

### 2. Exploratory data analysis

+ Univariate
+ Bivariate
+ Multivariate
### 3. Data preprocessing for Machine Learning Modeling

+ Converted categorical data to numerical data using one-hot encoding
+ Scaled feature values to mitigate the effect of outliers
+ Performed feature selection based on importance score

### 4. Model Training

+ Treated imbalanced data
+ Split data into training and test data
+ Trained Random Forest model on training data
+ Tested the model's performance on the test data

### 5. Model Evaluation and Improvement

Evaluated the model's performance on the training and test data using evaluation metrics

## Key Insights

### A.
<img width="985" alt="Screenshot 2025-03-08 at 12 58 37 PM" src="https://github.com/user-attachments/assets/1429769e-a861-4e9a-8f55-e255241621a6" />


### B.
<img width="986" alt="Screenshot 2025-03-08 at 12 59 27 PM" src="https://github.com/user-attachments/assets/f6ac8bf8-0eeb-4d3a-8665-9e1671ccbc29" />


### C.
<img width="940" alt="Screenshot 2025-03-08 at 1 22 31 PM" src="https://github.com/user-attachments/assets/019ce851-8162-4018-8f17-58f08a78bf0c" />

### D.
<img width="983" alt="Screenshot 2025-03-08 at 1 23 22 PM" src="https://github.com/user-attachments/assets/dd290c12-f272-4d9f-a8c6-42a17fb33da7" />



### E.
<img width="976" alt="Screenshot 2025-03-08 at 1 11 38 PM" src="https://github.com/user-attachments/assets/bd60c2d0-a9e8-4710-b758-d2bfb8d24dd4" />

### F.
<img width="984" alt="Screenshot 2025-03-08 at 1 27 02 PM" src="https://github.com/user-attachments/assets/a8f171c7-d288-480e-b531-b7be8ee5c23f" />


### G.
<img width="986" alt="Screenshot 2025-03-08 at 1 31 18 PM" src="https://github.com/user-attachments/assets/a6f4d0c8-9dbc-4a5b-831f-23000d72afb7" />


### H.
<img width="985" alt="Screenshot 2025-03-08 at 1 32 59 PM" src="https://github.com/user-attachments/assets/13ea9d24-95ad-41a8-b407-3ea60acf934b" />


### I.
<img width="971" alt="Screenshot 2025-03-08 at 1 34 51 PM" src="https://github.com/user-attachments/assets/5f3e33da-0b92-4c71-80c4-2c090b71da04" />


### J. Correlation between heart disease and associated factors
<img width="983" alt="Screenshot 2025-03-08 at 1 43 17 PM" src="https://github.com/user-attachments/assets/da0eb4fb-0dca-4682-a8a9-d8239ac25f00" />



### K.

<img width="992" alt="Screenshot 2025-03-08 at 1 42 24 PM" src="https://github.com/user-attachments/assets/93485f1a-6aa4-4b47-bdbd-78f6c998aad1" />


### L.
<img width="988" alt="Screenshot 2025-03-08 at 1 46 01 PM" src="https://github.com/user-attachments/assets/e11a4021-6ee6-48ae-b119-b6928995f1d6" />


### M.
<img width="978" alt="Screenshot 2025-03-08 at 1 47 15 PM" src="https://github.com/user-attachments/assets/bc242498-eaef-4d42-b4e1-c0f33327b58f" />


## Recommendations

1. The tuned Random Forest classification model is suitable for predicting the health status of patients in a hospital setting, given its high performance on both training and test data and its comparable predictive performance between the two classes (patients with and without heart disease)

2. Selection of features based on importance scores helps the model to focus on relevant features influencing the predicted outcome

3. Health investigations for heart disease should be routinely done as patients grow older and more among males, as this will help to prevent or arrest heart disease before it escalates or advances, in turn reducing healthcare costs

4. As investigations are performed, healthcare providers should seek out more potential influencers of heart disease among the population. These factors, when fed into the model, may provide better insights into more definite factors causing or associated with heart disease onset and progression
