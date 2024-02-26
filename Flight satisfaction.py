#!/usr/bin/env python
# coding: utf-8

# In[257]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[258]:


data=pd.read_csv("Invistico_Airline.csv")
data.sample(10)


# In[259]:


data["Class"].value_counts()


# In[260]:


clean=data.dropna()
clean.info()


# In[261]:


# convert categorical data into numericals
from sklearn.preprocessing import OneHotEncoder
clean["satisfaction"]=OneHotEncoder(drop='first').fit_transform(clean[["satisfaction"]]).toarray()
clean["Customer Type"]=OneHotEncoder(drop='first').fit_transform(clean[["Customer Type"]]).toarray()
clean["Type of Travel"]=OneHotEncoder(drop='first').fit_transform(clean[["Type of Travel"]]).toarray()
ordinal_mapping = {'Business': 3, 'Eco': 2, 'Eco Plus': 1}
clean['Class'] = clean['Class'].map(ordinal_mapping)


# In[262]:


clean['Class'].value_counts()


# In[252]:


clean.shape


# In[184]:


sample_data=clean.sample(n=50)


# In[25]:


sns.pairplot(sample_data,hue="satisfaction")


# In[263]:


import matplotlib.pyplot as plt
corr_matrix = clean.corr().abs()
# Create a heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Correlation Matrix (Absolute Values)')
plt.show()


# In[264]:


X = clean.drop("satisfaction", axis=1)
y = clean["satisfaction"]


# In[265]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Assuming 'clean' is your DataFrame

X = clean.drop("satisfaction", axis=1)
y = clean["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost":xgb.XGBClassifier()
}



for model_name, model in models.items():
    model.fit(X_train, y_train)

    # Prediction
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    # Training performance
    model_train_accuracy = accuracy_score(y_train, y_train_predict)
    model_train_precision = precision_score(y_train, y_train_predict, average='weighted')
    model_train_accuracy_f1score = f1_score(y_train, y_train_predict, average='weighted')
    model_train_accuracy_recall = recall_score(y_train, y_train_predict, average='weighted')

    # Testing performance
    model_test_accuracy = accuracy_score(y_test, y_test_predict)
    model_test_precision = precision_score(y_test, y_test_predict, average='weighted')
    model_test_accuracy_f1score = f1_score(y_test, y_test_predict, average='weighted')
    model_test_accuracy_recall = recall_score(y_test, y_test_predict, average='weighted')

    # Print model name
    print(f'Model: {model_name}\n')

    # Print training performance
    print('Training Performance:')
    print('   Accuracy: {:.4f}'.format(model_train_accuracy))
    print('   Precision: {:.4f}'.format(model_train_precision))
    print('   F1-score: {:.4f}'.format(model_train_accuracy_f1score))
    print('   Recall: {:.4f}'.format(model_train_accuracy_recall))

    print('\n---------------------------------------------------------------\n')

    # Print testing performance
    print('Testing Performance:')
    print('   Accuracy: {:.4f}'.format(model_test_accuracy))
    print('   Precision: {:.4f}'.format(model_test_precision))
    print('   F1-score: {:.4f}'.format(model_test_accuracy_f1score))
    print('   Recall: {:.4f}'.format(model_test_accuracy_recall))

    print('\n================================================================\n')

  


# In[266]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Initialize a plot
plt.figure(figsize=(10, 8))

# Plot ROC curve for each model
for model_name, model in models.items():
    # Get predicted probabilities for the positive class
    y_train_probs = model.predict_proba(X_train)[:, 1]
    y_test_probs = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC for training set
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_probs)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Compute ROC curve and AUC for testing set
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_probs)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Plot ROC curve for training set
    plt.plot(fpr_train, tpr_train, label=f'{model_name} (Train) - AUC = {roc_auc_train:.2f}')

    # Plot ROC curve for testing set
    plt.plot(fpr_test, tpr_test, label=f'{model_name} (Test) - AUC = {roc_auc_test:.2f}', linestyle='dashed')

# Plot the diagonal line representing random guessing
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Guessing')

# Set plot labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend()

# Show the plot
plt.show()


# In[267]:


# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(15, 5))

# Plot ROC curve for each model
for (model_name, model), ax in zip(models.items(), axes):
    # Get predicted probabilities for the positive class
    y_train_probs = model.predict_proba(X_train)[:, 1]
    y_test_probs = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC for training set
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_probs)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Compute ROC curve and AUC for testing set
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_probs)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Plot ROC curve for training set
    ax.plot(fpr_train, tpr_train, label=f'Train - AUC = {roc_auc_train:.2f}')

    # Plot ROC curve for testing set
    ax.plot(fpr_test, tpr_test, label=f'Test - AUC = {roc_auc_test:.2f}', linestyle='dashed')

    # Set plot labels and title
    ax.set_title(model_name)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:




