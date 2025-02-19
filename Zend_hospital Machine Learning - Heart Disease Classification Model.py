#!/usr/bin/env python
# coding: utf-8

# Heart Disease Classification: Exploratory Data Analysis (EDA) and Building a Machine Learning Model

# Project Breakdown

# Business Objective: The goal of this project is to perform indepth EDA to uncover trends and patterns, and also identify factors influencing heart disease development using historical data and specific features.

# Dataset: The dataset contains information about patients attending Zend hospital including demographic/socioeconomic and health behavioural data, having the following features:
# id: Unique identifier for each patient
# Gender: Gender of the patient (Male/Female)
# age: Age of the patient
# heart_disease: Account balance of the customer
# ever_married: Patient's marital status
# work_type: Sector patient is employed in
# Residence_type: Whether the patient lives in an urban or rural area
# avg_glucose_level: Blood glucose level
# bmi: Body mass index
# smoking_status: Whether patient has is an active or former smoker

# Project Goals:
# Conduct exploratory data analysis to uncover trends and patterns associated with heart disease in patients, and identify factors influencing the health condition
# Build a machine learning model to classify heart disease risk

# Analysis Framework:
# Exploratory data analysis following data loading, including univariate, bivariate and multivariate analysis
# Data preprocessing, including treating missing values and outliers, converting categorical data to numerical data using on-hot encoding,   scaling feature values, feature selection based on importance
  
# Model Selection and Training:
# Treat imbalanced data
# Split data into training and test data
# Train model (Random Forest) on training data
# Test the model's performance on the test data
  
# Model Evaluation and Improvement**:
# Evaluate model's performance on the training and test data using evaluation metrics. The model will be evaluated using standard classification metrics such as accuracy, precision, recall, and F1-score. The aim is to build a model with high predictive accuracy and minimize the number of false positives and false negatives.
  
# Key Insights, Recommendation and Conclusion

# Exploratory data analysis and machine learning model will help the hospital management to identify key factors influencing heart disease    risk among their patient population and take concerted/directed actions to mitigate the problem.

# Import python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
patient_df = pd.read_csv('./Heart_disease_prediction.csv')

print(patient_df)

print(patient_df.tail())

# Data Preprocessing for EDA

# Check for missiing values
print(patient_df.isnull().sum())

# Check range of BMI values
print(patient_df.min(axis = 0)['bmi'])
print(patient_df.max(axis = 0)['bmi'])


# Replace missing values with the median bmi
patient_df['bmi'] = patient_df['bmi'].fillna(patient_df['bmi'].median())

# Check for missiing values again
print(patient_df.isnull().sum())

# Check for duplicates
print(patient_df.duplicated().any())


# EDA - Exploratory Data Analysis


print(patient_df.info())

# Check number of rows and columns
print(patient_df.shape)

# Perform summary statistics
print(patient_df.describe())

# Extract column names
print(patient_df.columns)

# delete patient id, not needed
patient_df = patient_df.drop('id', axis=1)


print(patient_df.head())

# Check the distribution using histogram
sns.histplot(patient_df['age'])
plt.show()

# Boxplot to see bmi
sns.boxplot(data = patient_df, x = 'bmi');
plt.show()

# Boxplot to see age
sns.boxplot(data = patient_df, x = 'age');
plt.show()

# Check the total number of patients with or without heart disease
print(patient_df['heart_disease'].value_counts())

# Plotting the countplot with data label
ax = sns.countplot(x = str('heart_disease'), data = patient_df, hue = 'heart_disease')

# Adding data labels to the countplot
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Adding labels and title
ax.set_xlabel('heart_disease')
ax.set_ylabel('Count')
ax.set_title('Count of patients with heart disease')

# Display the plot
plt.show()

print(patient_df.head())

def plot_bar(category):
    plt.figure()
    counts = patient_df[category].value_counts()
    counts.plot(kind='bar')
    plt.title(f"{category} Distribution")
    plt.xlabel(category)
    plt.ylabel("Count")
    
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.show()


category1 = ["Gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

for c in category1:
    plot_bar(c)


sns.boxplot(x='heart_disease', y='age', data = patient_df, hue = 'heart_disease')
plt.show()

sns.boxplot(x='heart_disease', y='avg_glucose_level', data = patient_df, hue='heart_disease')
plt.show()

sns.boxplot(x='heart_disease', y='bmi', data = patient_df, hue='heart_disease')
plt.show()

sns.barplot(x='heart_disease', y='bmi', data = patient_df, hue='Gender')
plt.show()

# Plotting the countplot with labels
ax = sns.countplot(x='Gender', data=patient_df, hue='heart_disease')

# Adding labels and title
ax.set_xlabel('Number of Patients')
ax.set_ylabel('Count')
ax.set_title('Count of patients with heart disease by gender')

# Adding legend
ax.legend(title='heart_disease', labels=['No', 'Yes'])

# Display the plot
plt.show()

# Plotting the countplot with labels
ax = sns.countplot(x='ever_married', data=patient_df, hue='heart_disease')

# Adding labels and title
ax.set_xlabel('Number of Patients')
ax.set_ylabel('Count')
ax.set_title('Count of Patients with heart disease by marital status')

# Adding legend
ax.legend(title='heart_disease', labels=['No', 'Yes'])

# Display the plot
plt.show()

# Plotting the countplot with labels
ax = sns.countplot(x='smoking_status', data=patient_df, hue='heart_disease')

# Adding labels and title
ax.set_xlabel('Number of Patients')
ax.set_ylabel('Count')
ax.set_title('Count of Patients with heart disease by smoking status')

# Adding legend
ax.legend(title='heart_disease', labels=['No', 'Yes'])

# Display the plot
plt.show()

# Check if any two variables are correlated or related
sns.scatterplot(y = 'bmi', x = 'age', hue='heart_disease', data = patient_df)
plt.show()

# Check for correlation among numeric factors
corr_mat = patient_df.corr(numeric_only = True)
# color: cmap="YlGnBu"
sns.set()
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(corr_mat, linewidths=.5, ax=ax, cmap='YlGnBu', annot = True)
plt.show()

# Check categorical columns that need to be converted to numeric columns. O is AKA object
cat_features = [x for x in patient_df.columns if patient_df[x].dtype == 'O']
print(cat_features)

# Performs one-hot encoding: Convert categorical features in the dataframe to numeric variable
patient_df = pd.get_dummies(patient_df, cat_features, dtype =  int)

# validate your data
patient_df.head()

#import ML libraries
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler # used to perform feature scaling
from imblearn.over_sampling import SMOTE #creates synthetic samples for the minority class to balance the class distribution
from sklearn.feature_selection import SelectKBest, f_classif # This function selects the top k features based on a scoring function. 
from sklearn.metrics import accuracy_score, classification_report #provides a more detailed evaluation report for classification models, including precision, recall, F1-score, and support for each class.
from sklearn.ensemble import RandomForestClassifier
# library to build model
from sklearn.tree import DecisionTreeClassifier

# library to tune model
from sklearn.model_selection import GridSearchCV

# library to evaluate model performance
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# libraries to evaluate the model
from sklearn.metrics import f1_score, make_scorer

# Define independent variable
X = patient_df.drop(['heart_disease'], axis=1) #features
# Define the label or dependent variable
y = patient_df['heart_disease'] # target variable or label

# Perform feature scaling to standardize the features & ensure they have similar scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X, y)
selected_indices = selector.get_support(indices=True)
selected_features = patient_df.columns[selected_indices]
f_scores = selector.scores_[selected_indices]  # Extract f_scores only for selected features

# Visualization
plt.figure(figsize=(12, 6))
plt.bar(selected_features, f_scores, color='skyblue')
plt.xlabel('Selected Features')
plt.ylabel('Importance Score (F-statistic)')
plt.title('Feature Importance After Selection (Top 10)')
plt.xticks(rotation=45, ha='right')  # Rotate feature names for readability
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Print the selected feature names
selected_features = patient_df.columns[selected_indices]
print("Selected Features:")
for feature in selected_features:
    print(feature)

# check the total number of counts for target variable
patient_df['heart_disease'].value_counts()

# Perform data balancing using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Before balancing:")
print(y.value_counts())

print("After balancing:")
print(y_resampled.value_counts())


# Model Building - RandomForest Classifier

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42):

model = RandomForestClassifier(class_weight="balanced", random_state=1)

# train the model using traing data
model.fit(X_train, y_train)

def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

# confusion matrix for train data
confusion_matrix_sklearn(model, X_train, y_train)

# evaluating the model performance on the training data
model_train_predictions = model.predict(X_train)
model_train_score = f1_score(y_train, model_train_predictions)

print("Model Score on Train Data:", np.round(100*model_train_score, 2))

# confusion matrix for test data
confusion_matrix_sklearn(model, X_test, y_test)

# evaluating the model performance on the test data
model_test_predictions = model.predict(X_test)
model_test_score = f1_score(y_test, model_test_predictions)

print("Model Score on Test Data:", np.round(100*model_test_score, 2))

# Model Tuning

# choosing the type of model
dummy_model = RandomForestClassifier(class_weight='balanced', random_state=1)

# defining the grid of parameters of the AI model to choose from
parameters = {
    'max_depth': [3,4,5,6],
    'min_samples_leaf': np.arange(5,25,5),
    'max_features': [0.6,0.7,0.8],
    'n_estimators': np.arange(50,250,50)
}

# defining the model score on which we want to compare parameter combinations
scorer = make_scorer(f1_score)

# used for hyperparameter tuning.
grid_obj = GridSearchCV(dummy_model, parameters, scoring=scorer, cv=5, n_jobs=-2)
grid_obj = grid_obj.fit(X_train, y_train)

# selecting the best combination of parameters for the model to create a new model
tuned_model = grid_obj.best_estimator_

# Training on Tuned-model

# training the new tuned model
tuned_model.fit(X_train, y_train)

# evaluating the model performance on the train data
tuned_model_train_predictions = tuned_model.predict(X_train)
tuned_model_train_score = f1_score(y_train, tuned_model_train_predictions)

print("Model Score on Train Data:", np.round(100*tuned_model_train_score, 2))


# Model Testing

# evaluating the model performance on the test data
tuned_model_test_predictions = tuned_model.predict(X_test)
tuned_model_test_score = f1_score(y_test, tuned_model_test_predictions)

print("Model Score on Test Data:", np.round(100*tuned_model_test_score, 2))

# Make Prediction with Tuned Model that generalizes well

# Make predictions on the test set
y_pred = tuned_model.predict(X_test)

print("Predicted labels:")
print(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report}")