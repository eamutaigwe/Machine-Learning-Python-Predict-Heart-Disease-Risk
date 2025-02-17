#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Classification: Exploratory Data Analysis (EDA) and Building a Machine Learning Model

# ### Project Breakdown
# 
# - **Business Objective:** The goal of this project is to perform indepth EDA to uncover trends and patterns, and also identify factors influencing heart disease development using historical data and specific features.
# 
# - **Dataset:** The dataset contains information about patients attending Zend hospital including demographic/socioeconomic and health behavioural data, having the following features:
# - *id*: Unique identifier for each patient
# - *Gender*: Gender of the patient (Male/Female)
# - *age*: Age of the patient
# - *heart_disease*: Account balance of the customer
# - *ever_married*: Patient's marital status
# - *work_type*: Sector patient is employed in
# - *Residence_type*: Whether the patient lives in an urban or rural area
# - *avg_glucose_level*: Blood glucose level
# - *bmi*: Body mass index
# - *smoking_status*: Whether patient has is an active or former smoker
# 
# - **Project Goals:**
#  - Conduct exploratory data analysis to uncover trends and patterns associated with heart disease in patients, and identify factors influencing the health condition
#  - Build a machine learning model to classify heart disease risk
# 
# - **Analysis Framework:**
#  - Exploratory data analysis following data loading, including univariate, bivariate and multivariate analysis
#  - Data preprocessing, including treating missing values and outliers, converting categorical data to numerical data using on-hot encoding,   scaling feature values, feature selection based on importance
#   
# - **Model Selection and Training:**
#  - Treat imbalanced data
#  - Split data into training and test data
#  - Train model (Random Forest) on training data
#  - Test the model's performance on the test data
#   
# - **Model Evaluation and Improvement**:
#  - Evaluate model's performance on the training and test data using evaluation metrics. The model will be evaluated using standard classification metrics such as accuracy, precision, recall, and F1-score. The aim is to build a model with high predictive accuracy and minimize the number of false positives and false negatives.
#    
#  - **Key Insights, Recommendation and Conclusion**
# 
# - Exploratory data analysis and machine learning model will help the hospital management to identify key factors influencing heart disease    risk among their patient population and take concerted/directed actions to mitigate the problem.

# In[2]:


# Import python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[167]:


# Load data
patient_df = pd.read_csv('./Heart_disease_prediction.csv')


# In[169]:


patient_df


# In[171]:


patient_df.tail()


# ### Data Preprocessing for EDA

# In[9]:


# Check for missiing values
patient_df.isnull().sum()


# ### **Insight**
# 581 BMI values are missing.

# In[12]:


# Check range of BMI values
print(patient_df.min(axis = 0)['bmi'])
print(patient_df.max(axis = 0)['bmi'])


# ### **Insight**
# I will fill the missing values with the median BMI value, as the range between maximum and minimum BMI values is high and will influence the mean value.

# In[15]:


# Replace missing values with the median bmi
patient_df['bmi'] = patient_df['bmi'].fillna(patient_df['bmi'].median())


# In[17]:


# Check for missiing values again
patient_df.isnull().sum()


# In[19]:


# Check for duplicates
patient_df.duplicated().any()


# ### EDA - Exploratory Data Analysis

# In[22]:


patient_df.info()


# In[24]:


# Check number of rows and columns
patient_df.shape


# In[26]:


# Perform summary statistics
patient_df.describe()


# In[28]:


# Extract column names
patient_df.columns


# In[30]:


# delete patient id, not needed
patient_df = patient_df.drop('id', axis=1)


# In[32]:


patient_df.head()


# In[34]:


# Check the distribution using histogram
sns.histplot(patient_df['age'])


# In[36]:


# Boxplot to see bmi
sns.boxplot(data = patient_df, x = 'bmi');


# In[38]:


# Boxplot to see age
sns.boxplot(data = patient_df, x = 'age');


# ### **Insight**
# 
# - About 25% of the patients fall below approximately 25 years of age, 50% below 45 years, and 75% below 60 years. 
# - The remaining 25% fall below approximately 80years.

# In[41]:


# Check the total number of patients with or without heart disease
patient_df['heart_disease'].value_counts()


# In[43]:


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


# ### **Insight**
# 
# 4.9% of the patient population had heart disease. The data does not justify the claim that heart disease was on the rise at Zend hospital, although that may have been the case considering past records of heart disease at the facility.

# In[46]:


patient_df.head()


# In[50]:


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


# ### **Insights**
# 
# - There are many more female than male patients
# - There are almost 2 times more married patients than unmarried
# - The highest number of patients (more than 50%) work in the private sector
# - Patients reside equally in rural and urban areas
# - The highest number of patients have a history of smoking but have quit smoking. The next highest number of patients have never smoked.
# 

# In[53]:


sns.boxplot(x='heart_disease', y='age', data = patient_df, hue = 'heart_disease')
plt.show()


# ### **Insight**
# 
# - Most of the patients with heart disease are between the ages of 60 and 80 years suggesting that age is a significant factor influencing heart disease risk

# In[56]:


sns.boxplot(x='heart_disease', y='avg_glucose_level', data = patient_df, hue='heart_disease')
plt.show()


# ### **Insight**
# 
# - The patient group with heart disease had higher blood glucose levels than those without the disease

# In[59]:


sns.boxplot(x='heart_disease', y='bmi', data = patient_df, hue='heart_disease')
plt.show()


# In[61]:


sns.barplot(x='heart_disease', y='bmi', data = patient_df, hue='Gender')
plt.show()


# ### **Insight**
# 
# - The patients with heart disease also had higher body mass index values that those without, albeit the difference is not huge.
# - In addition, investigating the distribution of bmi by sex among patients with and without heart disease shows that both male and female patients without heart disease had similar distribution of BMI values. A similar result was observed in the heart disease group.

# In[64]:


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


# ### **Insights**
# 
# - Although the proportion of female patients is much higher than their male counterparts, more males have heart disease than females. 
# - It would be interesting to look into more factors peculiar to males that could have contributed to the higher incidence of heart disease among them.

# In[67]:


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


# ### **Insights**
# 
# - The number of married patients with heart disease is much higher than the number of unmarried patients with the disease. However, this could be due to the much higher number of married versus unmarried patients. 

# In[70]:


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


# ### **Insights**
# 
# - Due to the imbalance in the number of patients in the three smoking status categories, it may not be statistically accurate to compare the proportion of patients in each group with heart disease.

# In[75]:


# Check if any two variables are correlated or related
sns.scatterplot(y = 'bmi', x = 'age', hue='heart_disease', data = patient_df)
plt.show()


# ### **Insight**
# 
# - As shown in this plot, BMI does not show a strong correlation with age. However, it is seen that a majority of the patients with heart disease are in the older category

# In[77]:


# Check for correlation among numeric factors
corr_mat = patient_df.corr(numeric_only = True)
# color: cmap="YlGnBu"
sns.set()
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(corr_mat, linewidths=.5, ax=ax, cmap='YlGnBu', annot = True)
plt.show()


# ### **Insights**
# 
# - There is positive correlation between heart disease and age, average glucose level and BMI. However, BMI does not have high correlation with heart disease. Age and BMI are highly correlated. 

# ## Data Preprocessing for Machine Learning

# ### One-hot Encoding

# In[81]:


# Check categorical columns that need to be converted to numeric columns. O is AKA object
cat_features = [x for x in patient_df.columns if patient_df[x].dtype == 'O']
cat_features


# In[83]:


# Performs one-hot encoding: Convert categorical features in the dataframe to numeric variable
patient_df = pd.get_dummies(patient_df, cat_features, dtype =  int)


# In[85]:


# validate your data
patient_df.head()


# ### Import Machine Learning Libraries

# In[88]:


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


# In[90]:


# Define independent variable
X = patient_df.drop(['heart_disease'], axis=1) #features
# Define the label or dependent variable
y = patient_df['heart_disease'] # target variable or label


# In[92]:


# Perform feature scaling to standardize the features & ensure they have similar scale
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ### Feature Selection

# In[95]:


# Feature selection
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X, y)
selected_indices = selector.get_support(indices=True)
selected_features = patient_df.columns[selected_indices]
f_scores = selector.scores_[selected_indices]  # Extract f_scores only for selected features


# In[97]:


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


# In[99]:


# Print the selected feature names
selected_features = patient_df.columns[selected_indices]
print("Selected Features:")
for feature in selected_features:
    print(feature)


# # Building Machine Learning Model (Classification)

# ## Treat Imbalance in target variable using SMOTE Technique

# In[103]:


# check the total number of counts for target variable
patient_df['heart_disease'].value_counts()


# In[105]:


# Perform data balancing using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)


# In[107]:


print("Before balancing:")
print(y.value_counts())

print("After balancing:")
print(y_resampled.value_counts())


# # Model Building - RandomForest Classifier

# In[110]:


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# In[112]:


model = RandomForestClassifier(class_weight="balanced", random_state=1)


# In[114]:


# train the model using traing data
model.fit(X_train, y_train)


# In[115]:


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


# In[118]:


# confusion matrix for train data
confusion_matrix_sklearn(model, X_train, y_train)


# **Insights on Key Metrics**:
# 
# 1. True Positives (TP): 13811 instances were correctly predicted as belonging to class 1.
# 2. True Negatives (TN): 13768 instances were correctly predicted as belonging to class 0.
# 3. False Positives (FP): 0 instances were incorrectly predicted as class 1 when they actually belonged to class 0.
# 4. False Negatives (FN): 0 instances were incorrectly predicted as class 0 when they actually belonged to class 1.
# 
# **Overall Performance**:
# 
# The model accurately classified every instance, with no false positives or false negatives. This suggests possible overfitting.

# In[121]:


# evaluating the model performance on the training data
model_train_predictions = model.predict(X_train)
model_train_score = f1_score(y_train, model_train_predictions)

print("Model Score on Train Data:", np.round(100*model_train_score, 2))


# In[123]:


# confusion matrix for test data
confusion_matrix_sklearn(model, X_test, y_test)


# In[125]:


# evaluating the model performance on the test data
model_test_predictions = model.predict(X_test)
model_test_score = f1_score(y_test, model_test_predictions)

print("Model Score on Test Data:", np.round(100*model_test_score, 2))


# **Insights on Test Data**
# 
# 
# 1. True Negatives (TN): The model correctly predicted 3087 (44.78%) instances as negative (class 0).
# 2. True Positives (TP): The model correctly predicted 3306 (47.95%) instances as positive (class 1).
# 3. False Positives (FP): The model incorrectly predicted 384 (5.57%) instances as positive (class 1) when they were actually negative (class 0).
# 4. False Negatives (FN): The model incorrectly predicted 117 (1.70%) instances as negative (class 0) when they were actually positive (class 1).
# 
# **Overall Conclusion**:
# The model has high accuracy on the test data but did not perform as well as in the training data
# 
# 

# ## Key Insights from Model Evaluation
# 
# Randomforest classifier model scored 100% on the train data and 92.96% on the test data, respectively.
# 
# - This result shows that the model performs better on the train data than on the test data. Although the false positive instances constitute just 5.57%, interpretng the result as 384 patients erroneously diagnosed with heart disease isn't trivial. It could lead to allocating resources to cater for those additional patients, whereas those resources could be channeled into better causes. On the other hand, patients with heart disease predicted as not having the disease could result in not allocating enough resources toward their treatment. A much better model performance is important.
# 

# # Model Tuning

# In[130]:


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


# In[132]:


# selecting the best combination of parameters for the model to create a new model
tuned_model = grid_obj.best_estimator_


# # Training on Tuned-model

# In[135]:


# training the new tuned model
tuned_model.fit(X_train, y_train)


# In[137]:


# evaluating the model performance on the train data
tuned_model_train_predictions = tuned_model.predict(X_train)
tuned_model_train_score = f1_score(y_train, tuned_model_train_predictions)

print("Model Score on Train Data:", np.round(100*tuned_model_train_score, 2))


# # Model Testing

# In[139]:


# evaluating the model performance on the test data
tuned_model_test_predictions = tuned_model.predict(X_test)
tuned_model_test_score = f1_score(y_test, tuned_model_test_predictions)

print("Model Score on Test Data:", np.round(100*tuned_model_test_score, 2))


# ## Insights 
# 
# - Score of tuned model on training Data: 85.32%. The model performs well on training data.
# - Model Score on Test Data: 85.18%. This shows that the model also performs well on unseen data.
# 

# # Make Prediction with Tuned Model that generalizes well

# In[146]:


# Make predictions on the test set
y_pred = tuned_model.predict(X_test)


# In[148]:


print("Predicted labels:")
print(y_pred)


# In[150]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)


# In[152]:


# Print the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report}")


# #### Key Insights on the metrics performance on the prediction 
# 
# - The model accurately classified 84% of the patients as having heart disease, indicating strong performance.
# - Balanced Performance Across Classes: Apart from f1-score, precision and recall are inconsistent between both classes (0 and 1), suggesting the model may be biased towards one group. In healthcare settings, this model performance may have significant potential effect on patient health, and resource allocation and management.
# - F1-score as a Key Metric: The F1-score, a single measure of overall performance, has similar scores in both classes (0.83 and 0.85), suggesting good prediction of patients with and without heart disease.

# 
# ## Recommendation
# 
# The model is suitable for predicting health status of patients in a hospital setting, given its high performance on both training and test data, and its comparable predictive performance between the two classes (patients with and without heart disease).
# 
# Feature Importance Analysis: Selection of features based on importance scores helps the model to focus on relevant features influencing the predicted outcome. In this case, narrowing down the features to those with high importance in relation to heart disease. It is crucial to do predictions using tuned models employing only the features with high importance scores. This will improve the accuracy of the prediction.
# In this analysis, age, marital status, gender, work type, glucose level and a few other patient characteristics were selected as most importance for predicting heart disease risk.
# 
# ## Strategy to reduce heart disease 
# 
# - Health status investigations for heart disease should be routinely done as patients grow older and among males, as this will help to prevent and arrest heart disease before it escalates or advances, in turn reducing healthcare costs.
# - As investigations are performed, healthcare providers should seek out more potential influencers of heart disease among the population. These factors, when fed into the model, may provide better insights into more definite factors causing or associated with heart disease onset and progression.
# 
# 
# ## Conclusion
# 
# Narrowed-down feature selection based on feature importance analysis coupled with high prediction accuracy of the tuned model will provide specific and highly relevant insights for preventing and treating heart disease among the target population.
# 
# Furthermore, as more quality data are made available, they should be used to retrain the tuned model for deeper actionable insights and maintenance of the model's accuracy and predictive power.
