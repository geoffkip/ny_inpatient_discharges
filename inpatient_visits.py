#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 20:55:29 2018

@author: geoffrey.kip
"""

import pandas as pd
from os import chdir
import numpy as np
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from yellowbrick.regressor import ResidualsPlot
from sklearn import metrics

pd.options.display.float_format = "{:.2f}".format

wd = "/Users/geoffrey.kip/Projects/sparcs_data"
chdir(wd)

sparcs_df= pd.read_csv("/Users/geoffrey.kip/Projects/sparcs_data/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv")

sparcs_df["Length of Stay"] = pd.to_numeric(sparcs_df['Length of Stay'], errors="coerce")
columns_to_keep = ["APR Medical Surgical Description", "APR Risk of Mortality", "APR Severity of Illness Description",
              "Abortion Edit Indicator", "Age Group", "Emergency Department Indicator", "Ethnicity", "Gender",
              "Health Service Area", "Race","Type of Admission", "Length of Stay","Birth Weight",
               "Total Charges", "Total Costs","CCS Procedure Description"]
sparcs_df = sparcs_df[columns_to_keep]
sparcs_df.fillna(method="ffill",inplace=True)
print((sparcs_df.head()))

print(sparcs_df.shape)

print(sparcs_df.isnull().sum())

print(sparcs_df.dtypes)

print(sparcs_df.describe())

# Specifically Look at total charges
charges = sparcs_df["Total Charges"]
# TODO: Minimum total charges
minimum_charges = np.min(charges)

# TODO: Maximum total chages
maximum_charges = np.max(charges)

# TODO: Mean charges of the data
mean_charges = np.mean(charges)

# TODO: Median price of the data
median_charges = np.median(charges)

# TODO: Standard deviation of prices of the data
std_charges = np.std(charges)

# Show the calculated statistics
print ("Statistics for Ny Hospital Inpatient Discharges:\n")
print ("Minimum charge: ${:,.2f}".format(minimum_charges))
print ("Maximum charge: ${:,.2f}".format(maximum_charges))
print ("Mean charge: ${:,.2f}".format(mean_charges))
print ("Median charge ${:,.2f}".format(median_charges))
print ("Standard deviation of charge: ${:,.2f}".format(std_charges))


# Boxplot of prices to get a sense of the data

plt.title("Ny Inpatient Total charges")
plt.ylabel("Price (USD)")
plt.boxplot(charges)
plt.show()

# Boxplot of birth to get a sense of the data
birth_weight = sparcs_df["Birth Weight"]

plt.title("Ny Inpatient Birthweight")
plt.ylabel("Birthweight (grams)")
plt.hist(birth_weight)
plt.show()

print(sparcs_df["CCS Procedure Description"].value_counts())

sparcs_df["CCS Procedure Description"] = np.where(sparcs_df["CCS Procedure Description"] == "NO PROC" , 1, 
                             np.where(sparcs_df["CCS Procedure Description"] == "OTHER THERAPEUTIC PRCS", 2,
                             np.where(sparcs_df["CCS Procedure Description"] == "OT PRCS TO ASSIST DELIV", 3,
                             np.where(sparcs_df["CCS Procedure Description"] == "PROPHYLACTIC VAC/INOCUL", 4,
                             np.where(sparcs_df["CCS Procedure Description"] == "CESAREAN SECTION", 5,
                             np.where(sparcs_df["CCS Procedure Description"] == "RESP INTUB/MECH VENTIL", 6,
                             np.where(sparcs_df["CCS Procedure Description"] == "ALCO/DRUG REHAB/DETOX", 7, 
                             np.where(sparcs_df["CCS Procedure Description"] == "PSYCHO/PSYCHI EVAL/THER", 8,
                             np.where(sparcs_df["CCS Procedure Description"] == "CIRCUMCISION", 9,
                             np.where(sparcs_df["CCS Procedure Description"] == "OPHTHALM-/OT-OLOGIC DX", 10,
                             np.where(sparcs_df["CCS Procedure Description"] == "BLOOD TRANSFUSION", 11,
                             np.where(sparcs_df["CCS Procedure Description"] == "ARTHROPLASTY KNEE", 12, 
                             np.where(sparcs_df["CCS Procedure Description"] == "REPAIR CUR OBS LACERATN", 13,
                             np.where(sparcs_df["CCS Procedure Description"] == "HIP REPLACEMENT,TOT/PRT", 14,
                             np.where(sparcs_df["CCS Procedure Description"] == "UP GASTRO ENDOSC/BIOPSY", 15,
                             np.where(sparcs_df["CCS Procedure Description"] == "DX CARDIAC CATHETERIZTN", 16,
                             np.where(sparcs_df["CCS Procedure Description"] == "DX ULTRASOUND HEART", 17,
                             np.where(sparcs_df["CCS Procedure Description"] == "OTHER RESP THERAPY", 18,
                             np.where(sparcs_df["CCS Procedure Description"] == "SPINAL FUSION", 19,
                             np.where(sparcs_df["CCS Procedure Description"] == "HEMODIALYSIS", 20,
                             np.where(sparcs_df["CCS Procedure Description"] == "PERC TRANSLUM COR ANGIO", 21,
                             np.where(sparcs_df["CCS Procedure Description"] == "OT VASC CATH; NOT HEARTT", 22,
                             np.where(sparcs_df["CCS Procedure Description"] == "COMP AXIAL TOMOGR (CT)", 23,
                             np.where(sparcs_df["CCS Procedure Description"] == "PHYS THER EXER, MANIPUL", 24,
                             np.where(sparcs_df["CCS Procedure Description"] == "CHOLECYSTECTOMY/EXPLOR", 25,
                             np.where(sparcs_df["CCS Procedure Description"] == "OT OR PRCS VES NOT HEAD", 26,
                             np.where(sparcs_df["CCS Procedure Description"] == "OT DX PRC (INTERVW,EVAL", 27,
                             np.where(sparcs_df["CCS Procedure Description"] == "TRTMNT,FRAC HIP/FEMUR", 28,
                             np.where(sparcs_df["CCS Procedure Description"] == "APPENDECTOM", 29,
                             np.where(sparcs_df["CCS Procedure Description"] == "GASTRECTOMY; PART/TOTAL", 30,31))))))))))))))))))))))))))))))

print(sparcs_df["CCS Procedure Description"].value_counts())


cat_columns = ["APR Medical Surgical Description", "APR Risk of Mortality", "APR Severity of Illness Description",
              "Abortion Edit Indicator", "Age Group", "Emergency Department Indicator", "Ethnicity", "Gender",
              "Health Service Area", "Race","Type of Admission"]
for i, col in enumerate(sparcs_df[cat_columns]):
    plt.figure(i)
    sns.countplot(x=col, data=sparcs_df)
    

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(sparcs_df, x_vars=['Birth Weight','Total Charges','Total Costs'], y_vars='Length of Stay', size=7, aspect=0.7)

## Predict length of stay using Regression algorithms 

# Split data into features and target
X =  sparcs_df[sparcs_df.columns.difference(["Length of Stay", "CCS Procedure Description"])]
Y = sparcs_df["Length of Stay"]
print(X.shape)
print(Y.shape)

train_quant= X.select_dtypes(include=[np.float64,np.int64])
train_categorical= X.select_dtypes(include=[np.object])
train_categorical= pd.get_dummies(train_categorical)
train_categorical.head()
train_features= pd.concat([train_categorical, train_quant], axis=1)
X =  train_features
print(X.shape)

#Split data into training and test sets
test_size = 0.30
#validation_size=0.20
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
print (X_train.shape, Y_train.shape)
#print (X_validation.shape, Y_validation.shape)
print (X_test.shape, Y_test.shape)

seed = 7
scoring = 'neg_mean_squared_error'
# Evaluate training accuracy
models = []
models.append(('LR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Lasso', Lasso()))
models.append(('Elastic Net', ElasticNet()))
#models.append(('KNN',  KNeighborsRegressor()))
models.append(('Decision Tree', DecisionTreeRegressor))
#models.append(('Support Vector', SVR()))

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Compare Algorithms
fig = plt.figure(figsize=(10,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Fit KNN regressor
model =  KNeighborsRegressor()
model.fit(X_train, Y_train)

# Instantiate the linear model and visualizer
visualizer = ResidualsPlot(model)

visualizer.fit(X_train, Y_train)  # Fit the training data to the model
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
visualizer.poof()                 # Draw/show/poof the data

#Predict with algorithm
predictions = model.predict(X_test)

print(mean_absolute_error(Y_test,predictions))
print(mean_squared_error(Y_test,predictions))
print(r2_score(Y_test,predictions))

prediction_df= pd.DataFrame(predictions, columns=["prediction"])
real_df= pd.DataFrame(Y_test).reset_index(drop=True)
comparison_data= pd.merge(real_df , prediction_df, how='left', left_index=True, right_index=True)
comparison_data["Residual"] = comparison_data["Length of Stay"] - comparison_data["prediction"]

# Plot the residuals after fitting a linear model
sns.residplot(comparison_data["prediction"], comparison_data["Residual"], lowess=True, color="g")


## Predicting the procedure type using classification model

# Split data into features and target
Y = sparcs_df["CCS Procedure Description"]
print(X.shape)
print(Y.shape)

#Split data into training and test sets
test_size = 0.30
#validation_size=0.20
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
print (X_train.shape, Y_train.shape)
#print (X_validation.shape, Y_validation.shape)
print (X_test.shape, Y_test.shape)

seed = 7
scoring = 'f1'
# Evaluate training accuracy
models = []
#models.append(('XGBOOST', XGBClassifier()))
models.append(('LR', LogisticRegression()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('RANDOM FOREST', RandomForestClassifier()))
#models.append(('SVM', SVC()))
#models.append(('Gradient Boosting', GradientBoostingClassifier()))

results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

model =  DecisionTreeClassifier()
model.fit(X_train, Y_train)

feat_imp = pd.Series(model.feature_importances_, X_train.columns).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

best_features= feat_imp[feat_imp > 0]
best_features_columns= list(best_features.index)

predictions = model.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
score_test = metrics.f1_score(Y_test, predictions,
                              pos_label=list(set(Y_test)), average = None)
