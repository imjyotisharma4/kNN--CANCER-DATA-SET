#importing libraries
import os
import numpy as np
import pandas as pd

os.chdir("C:\\Users\\hp\\Datasets")

# Loading the data
FullRaw = pd.read_csv("cancerdata.csv")

# Missing Value Check
FullRaw.isnull().sum()

#Data types
FullRaw.dtypes

#dummy variable creation- Manually
FullRaw['diagnosis'].value_counts() # Majority Class is "B"
FullRaw['diagnosis'] = np.where(FullRaw['diagnosis'] == 'B', 0, 1)


# Drop 'id' column
FullRaw.drop(['id'], axis = 1, inplace = True)
FullRaw.shape


# Sampling
from sklearn.model_selection import train_test_split
Train, Test = train_test_split(FullRaw, train_size = 0.8, random_state = 123)

Train["diagnosis"].value_counts() 

depVar = 'diagnosis'
trainX = Train.drop([depVar], axis = 1).copy()
trainY = Train[depVar].copy()
testX = Test.drop([depVar], axis = 1).copy()
testY = Test[depVar].copy()

trainX.shape
testX.shape


# Standardization
from sklearn.preprocessing import StandardScaler

Train_Scaling = StandardScaler().fit(trainX)                # Train_Scaling contains means, std_dev of training dataset
trainX_Std = Train_Scaling.transform(trainX)                # This step standardizes (executes the formula) the train data
testX_Std  = Train_Scaling.transform(testX)                 # This step standardizes (executes the formula) the test data
trainX_Std[0]

# Add the column names to trainX_Std, testX_Std
trainX_Std= pd.DataFrame(trainX_Std, columns = trainX.columns)
testX_Std = pd.DataFrame(testX_Std, columns = testX.columns)


# Model building
from sklearn.neighbors import KNeighborsClassifier
M1 = KNeighborsClassifier().fit(trainX_Std, trainY)

# Model prediction
# Class Prediction

testPred = M1.predict(testX_Std)

# Create a df to store the model predictions
Test_Prob_Df = pd.DataFrame()
Test_Prob_Df['Class'] = testPred

# Probability Prediction 
Test_Prob = pd.DataFrame(M1.predict_proba(testX_Std))                #Returns probability estimates using test data
Test_Prob_Df = pd.concat([Test_Prob_Df, Test_Prob], axis = 1)


# Optional: For ROC Curve - Create a SINGLE column to capture the probabilities (of 0 and 1 class)



# Model evaluation

from sklearn.metrics import confusion_matrix, classification_report

Confusion_Mat = pd.DataFrame(confusion_matrix(testY, testPred))
Confusion_Mat

from sklearn.model_selection import GridSearchCV

myNN = range(1,14,2)                                   # list(range(1,14,2))
myP = range(1,4,1)                                    # list(range(1,4,1)) . This "p" is "h" in minkowski formula
my_param_grid = {'n_neighbors': myNN, 'p': myP}      # param_grid is a dictionary 

Grid_Search_Model = GridSearchCV(estimator = KNeighborsClassifier(), 
                     param_grid=my_param_grid,  
                     scoring='accuracy', 
                     cv=5,n_jobs=-1).fit(trainX_Std, trainY)


Grid_Search_Df = pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)
Grid_Search_Df.to_csv("GridSearchKNN.csv")


# sum(np.diagonal(Confusion_Mat))/testX.shape[0]*100   
# Accuracy
print(classification_report(testY, testPred))



# Grid Search CV

# Data Re-Distribution (Changing data when there are outliers present in some columns)
import seaborn as sns
import numpy as np

trainX.columns
columnsToConsider = ["perimeter_mean", "area_mean", "perimeter_worst", "area_worst"]


# Histogram using seaborn
sns.pairplot(trainX[columnsToConsider])


# Lets consider Area Mean and apply log transformation
trainX_Copy = trainX.copy()
trainX_Copy["area_mean"] = np.log(np.where(trainX_Copy["area_mean"] == 0, 1, trainX_Copy["area_mean"]))
trainX_Copy["perimeter_worst"] = np.log(np.where(trainX_Copy["perimeter_worst"] == 0, 1, trainX_Copy["perimeter_worst"]))
trainX_Copy["area_worst"] = np.sqrt(np.where(trainX_Copy["area_worst"] == 0, 1, trainX_Copy["area_worst"]))

testX_Copy = testX.copy()
testX_Copy["area_mean"] = np.log(np.where(testX_Copy["area_mean"] == 0, 1, testX_Copy["area_mean"]))
testX_Copy["perimeter_worst"] = np.log(np.where(testX_Copy["perimeter_worst"] == 0, 1, testX_Copy["perimeter_worst"]))
testX_Copy["area_worst"] = np.sqrt(np.where(testX_Copy["area_worst"] == 0, 1, testX_Copy["area_worst"]))


# Histogram using seaborn
sns.pairplot(trainX_Copy[columnsToConsider])


# Standardization
Train_Scaling = StandardScaler().fit(trainX_Copy) # Train_Scaling contains means, std_dev of training dataset
trainX_Std = Train_Scaling.transform(trainX_Copy) # This step standardizes the train data
testX_Std  = Train_Scaling.transform(testX_Copy) # This step standardizes the test data


# Add the column names to trainX_Std, testX_Std
trainX_Std = pd.DataFrame(trainX_Std, columns = trainX.columns)
testX_Std = pd.DataFrame(testX_Std, columns = testX.columns)


# Modeling
# Build model
M2 = KNeighborsClassifier().fit(trainX_Std, trainY)

# Class Prediction
testPred = M2.predict(testX_Std)


# Model evaluation
Confusion_Mat = confusion_matrix(testY, testPred)
Confusion_Mat

#sum(np.diagonal(Confusion_Mat))/testX.shape[0]*100 -- > Accuracy

print(classification_report(testY, testPred))

