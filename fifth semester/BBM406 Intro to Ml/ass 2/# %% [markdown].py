# %% [markdown]
# ### ## BBM 409 - Programming Assignment 2
# 
# * You can add as many cells as you want in-between each question.
# * Please add comments to your code to explain your work.  
# 
# * Please be careful about the order of runs of cells. Doing the homework, it is likely that you will be running the cells in different orders, however, they will be evaluated in the order they appear. Hence, please try running the cells in this order before submission to make sure they work.    
# * Please refer to the homework text for any implementation detail. You should also carefully review the steps explained here.
# * This document is also your report. Show your work.

# %% [markdown]
# ##  Insert personal information (name, surname, student id)

# %% [markdown]
# Salih Eren, Yüzbaşıoğlu, 2220356040

# %% [markdown]
# # 1. LOGISTIC REGRESSION TASK (40 points)

# %% [markdown]
# ### 1. Data Loading and Exploration

# %% [markdown]
# ##### Download the Bank Marketing dataset from https://drive.google.com/file/d/1t6QAtqfYLMhvv_XUnG4D_UsJcSwgF4an/view?usp=sharing  import other necessary libraries

# %%
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("./portuguese_bank_marketing_numeric_random_subsampled.csv", encoding="utf-8")

# %% [markdown]
# # Fixing data types

# %%
# check different values on job,contact,poutcome,education to see if float is needed
print(df['job'].value_counts())
print(df['contact'].value_counts())
print(df['poutcome'].value_counts())
print(df['education'].value_counts())

# %% [markdown]
# Turn job, contact, poutcome and education to integer type

# %%
df['job'] = df['job'].astype('category').cat.codes
df['contact'] = df['contact'].astype('category').cat.codes
df['poutcome'] = df['poutcome'].astype('category').cat.codes
df['education'] = df['education'].astype('category').cat.codes
df.dtypes

# %%
# check for nulls
print(df.isnull().sum())

# %%
print(df.describe())

# %% [markdown]
# ### 2. Calculate correlation between target variable 'y' and other features (5 points)

# %%
# correlation calculation
corr = df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=True,vmin=-1, vmax=1, center= 0, linewidths=3,cmap="YlGnBu")
plt.show()

# %%
# plot features
df.hist(figsize=(20,15))
plt.show()


# %% [markdown]
# # 1.1 Implementing Logistic Regression with most correlated 2 features

# %% [markdown]
# ###  Choose the two most correlated features with target feature 'y'

# %%
df['bias'] = 1
# replace 2 with 1 and 1 with 0 in y
df['y'] = df['y'].replace(1,0)
df['y'] = df['y'].replace(2,1)

# %%
# looking at above correlation figure, we can see that duration and poutcome have the highest correlation with y
X=df[['duration', 'poutcome','bias']]
y=df['y']

# %% [markdown]
# ###  * Define your logistic regression model as class without using any built-in libraries
# ### * Define necessary functions such as sigmoid, fit, predict  (10 points)

# %%
class LogisticRegression:
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))
    def __call__(self,x):
        return self.sigmoid(np.dot(x,self.w))
    def predict(self,x):
        return (self.sigmoid(np.dot(x,self.w)) >= 0.5).astype(int)
    def update(self,X,y,lr):
        # X.shape is (m,n) y.shape is (n,1) w.shape is (n,1)
        y_hat = self.sigmoid(np.dot(X,self.w))
        self.w = self.w - lr * np.dot(X.T, y_hat - y)    
    def fit(self,X,y,lr=0.001,epochs=100):
        y = y.reshape(-1,1)
        self.w = np.zeros((X.shape[1],1))
        for i in range(epochs):
            self.update(X,y,lr)
        

# %% [markdown]
# # Scale the features using standard scaler

# %%
# scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_nump = scaler.fit_transform(X)

# %% [markdown]
# Split the dataset into a training set and a validation set (80% training and 20% validation).

# %%
# split data
y_nump = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X_nump, y_nump, test_size=0.2, random_state=51)

# %% [markdown]
# * Initialize and train the custom logistic regression model

# %%
model = LogisticRegression()
model.fit(X_train,y_train)

# %% [markdown]
# * Make predictions on the validation set

# %%
y_pred = (model(X_test) >= 0.5).astype(int)

# %% [markdown]
# ### Evaluate the model's performance, print classification report and confusion matrix  (5 points)

# %%
def confusionMat(y_pred,y):
    # return numpy array of shape (2,2)
    ans = np.zeros((2,2))
    for i in range(len(y)):
        if y[i] == 1:
            if y_pred[i] == 1:
                ans[0][0] += 1
            else:
                ans[0][1] += 1
        else:
            if y_pred[i] == 1:
                ans[1][0] += 1
            else:
                ans[1][1] += 1
    return ans
def accuracy(confusionMat):
    return (confusionMat[0][0] + confusionMat[1][1]) / np.sum(confusionMat)
def precision(confusionMat):
    return confusionMat[0][0] / (confusionMat[0][0] + confusionMat[0][1])
def recall(confusionMat):   
    return confusionMat[0][0] / (confusionMat[0][0] + confusionMat[1][0])
def f1_score(confusionMat):
    return 2 * precision(confusionMat) * recall(confusionMat) / (precision(confusionMat) + recall(confusionMat))
confMat = confusionMat(y_pred,y_test)
print("Accuracy: ",accuracy(confMat))
print("Precision: ",precision(confMat))
print("Recall: ",recall(confMat))
print("F1 Score: ",f1_score(confMat))
print("Confusion Matrix: ",confMat)
# classfication report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### Print decision boundaries as in PA1 (5 points)

# %%
def plot_decision_boundary(X, y, model):
    # plot class 1 as blue points
    classOne = (y == 1).reshape(-1)
    classZero = (y == 0).reshape(-1)

    # plot the decision boundary
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape(-1,1), r2.reshape(-1,1)
    # horizontal stack vectors to create x1,x2 and bias
    grid = np.hstack((r1, r2, np.ones((r1.shape[0],1))))
    # make predictions
    # predictions = np.array([model(x) for x in grid])
    predictions = model.predict(grid)   
    # reshape the predictions back into a grid
    zz = predictions.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='cool',levels = 20, alpha=0.8)  

    # Scatter plots for each class with cool colors
    plt.scatter(X[classZero][:, 0], X[classZero][:, 1], color='blue', label='Class 0')  # Set color to purple for Class -1
    plt.scatter(X[classOne][:, 0], X[classOne][:, 1], color='purple', label='Class 1')      # Set color to blue for Class 1
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# %%
# plot the decision boundary
plot_decision_boundary(X_nump, y_nump, model)

# %% [markdown]
# # 1.2 Implementing Logistic Regression using all features.

# %% [markdown]
# * Redefine input and target variables. In this experiment, you will use all input features in the dataset.

# %%
# X is everything except y
X = df.drop(columns=['y'])
y = df['y']

# %% [markdown]
# * Scale the features using StandardScaler

# %%
scaler = StandardScaler()
X_nump = scaler.fit_transform(X)

# %% [markdown]
# * Split the dataset into a training set and a validation set (80% training and 20% validation).

# %%
y_nump = y.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X_nump, y_nump, test_size=0.2, random_state=51)

# %% [markdown]
# ### Initialize and train the custom logistic regression model.

# %%
fullModel = LogisticRegression()
fullModel.fit(x_train,y_train)

# %% [markdown]
# * Make predictions on the validation set

# %%
probs = fullModel(x_test)
y_pred = (probs >= 0.5).astype(int)

# %% [markdown]
# ### Evaluate the model's performance, print classification report and confusion matrix  (5 points)

# %%
# evaluate the model
confMat = confusionMat(y_pred,y_test)
print("Accuracy: ",accuracy(confMat))
print("Precision: ",precision(confMat))
print("Recall: ",recall(confMat))
print("F1 Score: ",f1_score(confMat))
print("Confusion Matrix: ",confMat)
# classfication report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### Briefly explain the impact of the number of features on the learning ability of the model. (5 points)

# %% [markdown]
# Looking at the classification report, we can see that adding more features has increased both accurary, precision and recall values. This is because the model has more information to learn from. But considering the amount features we have added and the how little the increase in the values are, model might be overfitting.

# %% [markdown]
# ### After completing the SVM and logistic regression tasks, the best results of the experiments with the SVM and Logistic regression models will be compared in a table. (5 points)

# %%


# %%


# %% [markdown]
# # 2. Support Vector Machine Task  (30 points)

# %% [markdown]
# * Define your SVM model using sklearn

# %%
# SVM
import sklearn.svm as svm
from sklearn.metrics import accuracy_score

# %% [markdown]
# ## 2.1 implementing svm with grid search cv using all features (10 points)

# %% [markdown]
# * Define features and target variable, you will use all features of dataset in this task

# %%
X = df.drop(columns=['y'])
y = df['y']

# %% [markdown]
# * Scale the features using StandardScaler

# %%
X_nump = scaler.fit_transform(X)
y_nump = y.to_numpy()

# %% [markdown]
# * Split the dataset into a training set and a validation set (80% training and 20% validation).

# %%
x_train, x_test, y_train, y_test = train_test_split(X_nump, y_nump, test_size=0.2, random_state=51)

# %% [markdown]
# #### Implement GridSearchCV  (5 points)

# %%
# grid search cv
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
grid.fit(x_train, y_train)

# %% [markdown]
# * Initialize the SVM classifier

# %%
# initlize the svm with best parameters
best_params = grid.best_params_
svmModel = svm.SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])

# %% [markdown]
# * Train the SVM classifier with the best parameters found from grid search
# 

# %%
# fit the model
svmModel.fit(x_train, y_train)

# %% [markdown]
# * Make predictions on the validation set using the best model
# 

# %%
# predict
y_pred = svmModel.predict(x_test)

# %% [markdown]
# #### Evaluate the model's performance, print classification report and confusion matrix and best parameters found from GridSearchCV  (5 points)

# %%
# evaluate the model
confMat = confusionMat(y_pred,y_test)
print("Accuracy: ",accuracy(confMat))
print("Precision: ",precision(confMat))
print("Recall: ",recall(confMat))
print("F1 Score: ",f1_score(confMat))
print("Confusion Matrix: ",confMat)
# classfication report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# best parameters
print("Best Parameters: ",best_params)

# %% [markdown]
# ## 2.2 implementing svm with most correlated 2 features (10 points)

# %% [markdown]
# #### Choose the two most correlated features with target feature 'y'

# %%
X = df[['duration', 'poutcome','bias']]
y = df['y']

# %% [markdown]
# * Scale the features using StandardScaler

# %%
# scale features
X_nump = scaler.fit_transform(X)
y_nump = y.to_numpy()

# %% [markdown]
# * Split the dataset into a training set and a validation set (80% training and 20% validation).

# %%
x_train, x_test, y_train, y_test = train_test_split(X_nump, y_nump, test_size=0.2, random_state=51)

# %% [markdown]
# *  Initialize the SVM classifier, assign 'C' and 'kernel' parameters from the best hyperparameters you found from GridSearchCV

# %%
# initlize the svm with best parameters
svmModel = svm.SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])

# %% [markdown]
# * Train the SVM classifier

# %%
# fit the model
svmModel.fit(x_train, y_train)

# %% [markdown]
# * Make predictions on the validation set

# %%
# predict
y_pred = svmModel.predict(x_test)

# %% [markdown]
# #### Evaluate the model's performance, print classification report and confusion matrix  (5 points)

# %%
# evaluate the model
confMat = confusionMat(y_pred,y_test)
print("Accuracy: ",accuracy(confMat))
print("Precision: ",precision(confMat))
print("Recall: ",recall(confMat))
print("F1 Score: ",f1_score(confMat))
print("Confusion Matrix: ",confMat)
# classfication report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# %% [markdown]
# ##### Visualize decision boundary and support vectors (5 points)

# %%
# plot the decision boundary
plot_decision_boundary(X_nump, y_nump, svmModel)

# %% [markdown]
# ## 2.3 implementing svm with least correlated 2 features (10 points)

# %% [markdown]
# #### Choose the two least correlated features with target feature 'y'

# %%
# job and month are least correlated
X = df[['job', 'month','bias']]
y = df['y']

# %% [markdown]
# * Scale the features using StandardScaler

# %%
X_nump = scaler.fit_transform(X)
y_nump = y.to_numpy()

# %% [markdown]
# * Split the dataset into a training set and a validation set (80% training and 20% validation).

# %%
x_train, x_test, y_train, y_test = train_test_split(X_nump, y_nump, test_size=0.2, random_state=51)

# %% [markdown]
# *  Initialize the SVM classifier, assign 'C' and 'kernel' parameters from the best hyperparameters you found from GridSearchCV

# %%
svmModel = svm.SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])

# %% [markdown]
# * Train the SVM classifier

# %%
# fit the model
svmModel.fit(x_train, y_train)

# %% [markdown]
# * Make predictions on the validation set

# %%
# predict
y_pred = svmModel.predict(x_test)

# %% [markdown]
# #### Evaluate the model's performance, print classification report and confusion matrix  (5 points)

# %%
# evaluate the model
confMat = confusionMat(y_pred,y_test)
print("Accuracy: ",accuracy(confMat))
print("Precision: ",precision(confMat))
print("Recall: ",recall(confMat))
print("F1 Score: ",f1_score(confMat))
print("Confusion Matrix: ",confMat)
# classfication report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# %% [markdown]
# ##### Visualize decision boundary and support vectors(5 points)

# %%
# plot the decision boundary
plot_decision_boundary(X_nump, y_nump, svmModel)

# %% [markdown]
# # 3. Decision Tree Task (30 points)

# %% [markdown]
# * Define your decision tree model using sklearn. Also you should define other necessary modules for visualize the decision tree

# %% [markdown]
# ### Download the dataset from https://drive.google.com/file/d/1D3peA-TzIqJqZDDKTlK0GQ7Ya6FIemFv/view?usp=sharing

# %% [markdown]
# ### import other necessary libraries

# %% [markdown]
# In this assignment, in multi-class classification task first you will explore the De-
# cision Tree algorithm by implementing it for a multi class classification task using
# the Weights Dataset. The dataset consists of data collected from 3,360 individuals
# aged 20 and over, comprising 9 (’BMI_CLASS’, ’UNIT_NUM’, ’STUB_NAME_NUM’,
# ’STUB_LABEL_NUM’, ’YEAR_NUM’, ’AGE_NUM’, ’ESTIMATE’, ’SE’, ’FLAG’) fea-
# tures and 6 discrete classes (’BMI_CLASS’ feature is the label feature). Each of the BMI
# classes is distributed equally. You will classify which BMI class a person belongs to among
# the six classes.

# %%
df=pd.read_csv("weights_bmi_6classes_updated.csv", encoding="utf-8")

# %% [markdown]
# * Define features and target variable, you will use all features of dataset in this task

# %%
X = df[['UNIT_NUM','STUB_NAME_NUM', 'STUB_LABEL_NUM', 'YEAR_NUM', 'AGE_NUM', 'ESTIMATE', 'SE', 'FLAG']]
y = df['BMI_CLASS']
X = X.replace('*',2)
X = X.replace('.',1)
X = X.replace('---',0)

# %% [markdown]
# * Split the dataset into a training set and a validation set (80% training and 20% validation).

# %%
X_nump = X.to_numpy()
y_nump = y.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X_nump, y_nump, test_size=0.2, random_state=5)

# %% [markdown]
# * Initialize the Decision Tree classifier

# %%
# decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=423,min_impurity_decrease=0.01,max_depth=15)

# %% [markdown]
# * Train the Decision Tree classifier

# %%
# fit the model
dt.fit(x_train, y_train)

# %% [markdown]
# * Make predictions on the validation set

# %%
# predict
y_pred = dt.predict(x_test)

# %% [markdown]
# #### Evaluate the model's performance, print classification report and confusion matrix  (10 points)

# %%
# evaluate the model
confMat = confusionMat(y_pred,y_test)
print("Accuracy: ",accuracy(confMat))
print("Precision: ",precision(confMat))
print("Recall: ",recall(confMat))
print("F1 Score: ",f1_score(confMat))
print("Confusion Matrix: ",confMat)
# classfication report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# %% [markdown]
# #### Visualize the Decision Tree, show clearly class number, gini value etc.  (10 points)
# 

# %%
# Visualize the Decision Tree, show clearly class number, gini value etc.
from sklearn.tree import plot_tree
plt.figure(figsize=(180,180))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=['0','1','2','3','4','5'])
plt.show()

# %% [markdown]
# ### Explain briefly the question. What is the role of gini in decision tree? (10 points)
# 
# Gini impurity is a measure similar to entropy that quantifies the disorder of a set of elements. It is calculated by summing the probability of an element being chosen times the probability of a mistake in categorizing that element. The Gini impurity is used to decide which feature to split on at each step in building the decision tree. The feature that results in the lowest Gini impurity is chosen as the splitting feature.


