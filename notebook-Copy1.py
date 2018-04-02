
# coding: utf-8

# Thank you for opening this script!
# 
# I have made all efforts to document each and every step involved in the prediction process so that this notebook acts as a good starting point for new Kagglers and new machine learning enthusiasts.
# 
# Please **upvote** this kernel so that it reaches the top of the chart and is easily locatable by new users. Your comments on how we can improve this kernel is welcome. Thanks.
# 
# My other exploratory studies can be accessed here :
# https://www.kaggle.com/sharmasanthosh/kernels
# ***
# ## Data statistics
# * Shape
# * Peek
# * Description
# * Skew
# 
# ## Transformation
# * Correction of skew
# 
# ## Data Interaction
# * Correlation
# * Scatter plot
# 
# ## Data Visualization
# * Box and density plots
# * Grouping of one hot encoded attributes
# 
# ## Data Preparation
# * One hot encoding of categorical data
# * Test-train split
# 
# ## Evaluation, prediction, and analysis
# * Linear Regression (Linear algo)
# * Ridge Regression (Linear algo)
# * LASSO Linear Regression (Linear algo)
# * Elastic Net Regression (Linear algo)
# * KNN (non-linear algo)
# * CART (non-linear algo)
# * SVM (Non-linear algo)
# * Bagged Decision Trees (Bagging)
# * Random Forest (Bagging)
# * Extra Trees (Bagging)
# * AdaBoost (Boosting)
# * Stochastic Gradient Boosting (Boosting)
# * MLP (Deep Learning)
# * XGBoost
# 
# ## Make Predictions
# ***

# ## Load raw data:
# 
# Information about all the attributes can be found here:
# 
# https://www.kaggle.com/c/allstate-claims-severity/data
# 
# Learning: 
# We need to predict the 'loss' based on the other attributes. Hence, this is a regression problem.

# In[1]:


# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

# Read raw data from the file

import pandas #provides data structures to quickly analyze data
#Since this code runs on Kaggle server, data can be accessed directly in the 'input' folder
#Read the train dataset
dataset = pandas.read_csv("input/pml_train.csv") 

#Read test dataset
dataset_test = pandas.read_csv("input/pml_test_features.csv")
#Save the id's for submission file
ID = dataset_test['id']
#Drop unnecessary columns
dataset_test.drop('id',axis=1,inplace=True)

#Print all rows and columns. Dont hide any
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

#Display the first five rows to get a feel of the data
print(dataset.head(5))

#Learning : cat1 to cat116 contain alphabets


# ## Data statistics
# * Shape

# In[2]:


# Size of the dataframe

print(dataset.shape)

# We can see that there are 188318 instances having 132 attributes

#Drop the first column 'id' since it just has serial numbers. Not useful in the prediction process.
dataset = dataset.iloc[:,1:]

#Learning : Data is loaded successfully as dimensions match the data description


# ## Data statistics
# * Description

# In[3]:


# Statistical description

print(dataset.describe())

# Learning :
# No attribute in continuous columns is missing as count is 188318 for all, all rows can be used
# No negative values are present. Tests such as chi2 can be used
# Statistics not displayed for categorical data


# ## Data statistics
# * Skew

# In[4]:


# Skewness of the distribution

print(dataset.skew())

# Values close to 0 show less ske
# loss shows the highest skew. Let us visualize it


# ## Data Visualization
# * Box and density plots

# In[5]:


# We will visualize all the continuous attributes using Violin Plot - a combination of box and density plots

import numpy

#import plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

#range of features considered
split = 116 

#number of features considered
size = 15

#create a dataframe with only continuous features
data=dataset.iloc[:,split:] 

#get the names of all the columns
cols=data.columns 

#Plot violin for all attributes in a 7x2 grid
n_cols = 2
n_rows = 7

#for i in range(n_rows):
#    fg,ax = plt.subplots(nrows=1,ncols=n_cols,figsize=(12, 8))
#    for j in range(n_cols):
#        sns.violinplot(y=cols[i*n_cols+j], data=dataset, ax=ax[j])


#cont1 has many values close to 0.5
#cont2 has a pattern where there a several spikes at specific points
#cont5 has many values near 0.3
#cont14 has a distinct pattern. 0.22 and 0.82 have a lot of concentration
#loss distribution must be converted to normal


# ## Data Transformation
# * Skew correction

# In[12]:


#log1p function applies log(1+x) to all elements of the column
dataset["loss"] = numpy.log1p(dataset["loss"])
#visualize the transformed column
sns.violinplot(data=dataset,y="loss")  
plt.show()

#Plot shows that skew is corrected to a large extent


# ## Data Interaction
# * Correlation

# In[13]:


# Correlation tells relation between two attributes.
# Correlation requires continous data. Hence, ignore categorical data

# Calculates pearson co-efficient for all combinations
data_corr = data.corr()

# Set the threshold to select only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))

# Strong correlation is observed between the following pairs
# This represents an opportunity to reduce the feature set through transformations such as PCA


# ## Data Visualization
# * Categorical attributes

# In[15]:


# Count of each label in each category

#names of all the columns
cols = dataset.columns

#Plot count plot for all attributes in a 29x4 grid
n_cols = 4
n_rows = 29
#for i in range(n_rows):
#    fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(12, 8))
#    for j in range(n_cols):
#        sns.countplot(x=cols[i*n_cols+j], data=dataset, ax=ax[j])

#cat1 to cat72 have only two labels A and B. In most of the cases, B has very few entries
#cat73 to cat 108 have more than two labels
#cat109 to cat116 have many labels

print(split)
i = 5
print(dataset[cols[i]].unique())
print(dataset_test[cols[i]].unique())


# ##Data Preparation
# * One Hot Encoding of categorical data

# In[55]:


import pandas

#cat1 to cat116 have strings. The ML algorithms we are going to study require numberical data
#One-hot encoding converts an attribute to a binary vector

#Variable to hold the list of variables for an attribute in the train and test data
labels = []


for i in range(0,split):
    train = dataset[cols[i]].unique()
    test = dataset_test[cols[i]].unique()
    labels.append(list(set(train) | set(test)))

#print(labels)

# del dataset_test

#Import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset.iloc[:,i])
    feature = feature.reshape(dataset.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
print()
encoded_cats = numpy.column_stack(cats)

# Print the shape of the encoded data
print(encoded_cats.shape)

#Concatenate encoded attributes with continuous attributes
dataset_encoded = numpy.concatenate((encoded_cats,dataset.iloc[:,split:].values),axis=1)
# del cats
# del feature
#del dataset
#del encoded_cats
print(dataset_encoded.shape)


# ##Data Preparation
# * Split into train and validation

# In[56]:


#get the number of rows and columns
r, c = dataset_encoded.shape

#create an array which has indexes of columns
i_cols = []
for i in range(0,c-1):
    i_cols.append(i)

#Y is the target column, X has the rest
X = dataset_encoded[:,0:(c-1)]
Y = dataset_encoded[:,(c-1)]
del dataset_encoded

#Validation chunk size
val_size = 0.1

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0

#Split the data into chunks
from sklearn import cross_validation
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)
#del X
#del Y

#All features
X_all = []

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae = []

#Scoring parameter
from sklearn.metrics import mean_absolute_error
# mean_absolute_error.__version__

#Add this version of X to the list 
n = "All"
#X_all.append([n, X_train,X_val,i_cols])
X_all.append([n, i_cols])
print(X.shape)
print(Y.shape)
print(X_train.shape)
print(X_val.shape)

#print(X_all.shape)
print([n, i_cols])


# ## Evaluation, prediction, and analysis
# * Linear Regression (Linear algo)

# In[59]:


#Evaluation of various combinations of LinearRegression

#Import the library
from sklearn.linear_model import LinearRegression

#uncomment the below lines if you want to run the algo
##Set the base model
model = LinearRegression(n_jobs=-1)
algo = "LR"

##Accuracy of the model using all features
cnt_loop = 0;
for name,i_cols_list in X_all:
    #print(name)
    #print(i_cols_list)
    model.fit(X_train[:,i_cols_list],Y_train)
    y_true = numpy.expm1(Y_val)
    y_pred = numpy.expm1(model.predict(X_val[:,i_cols_list]))
    print(y_true.shape)
    print(y_pred.shape)
    print(y_true)
    print(y_pred)
    
    
    result = mean_absolute_error(y_true, y_pred)
    print(result)
    mae.append(result)
    print(name + " %s" % result)
    print(cnt_loop)
    cnt_loop = cnt_loop+1
comb.append(algo)

#Result obtained after running the algo. Comment the below two lines if you want to run the algo
#mae.append(1278)
#comb.append("LR" )    

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#MAE achieved is 1278


# ## Evaluation, prediction, and analysis
# * LASSO Linear Regression (Linear algo)

# In[63]:


#Evaluation of various combinations of Lasso LinearRegression

#Import the library
from sklearn.linear_model import Lasso

#Add the alpha value to the below list if you want to run the algo
a_list = numpy.array([0.01, 0.05, 0.02, 0.005, 0.02, 0.001])
# alpha = 0.001
print(a_list)

for alpha in a_list:
    #Set the base model
    model = Lasso(alpha=alpha,random_state=seed)
    
    algo = "Lasso"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % alpha )

#Result obtained by running the algo for alpha=0.001    
#if (len(a_list)==0):
#    mae.append(1262.5)
#    comb.append("Lasso" + " %s" % 0.001 )
##Set figure size
plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
fig, ax = plt.subplots()
plt.plot(mae)
##Set the tick names to names of combinations
ax.set_xticks(range(len(comb)))
ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
plt.show()    

#High computation time
#Best estimated performance is 1262.5 for alpha = 0.001


# ## Evaluation, prediction, and analysis
# * Random Forest (Bagging)

# In[64]:


#Evaluation of various combinations of RandomForest

#Import the library
from sklearn.ensemble import RandomForestRegressor

#Add the n_estimators value to the below list if you want to run the algo
# dataset_encoded.shape: (188318, 1191) 
n_list = numpy.array([30, 50, 70, 100, 120, 150])

for n_estimators in n_list:
    #Set the base model
    model = RandomForestRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)
    
    algo = "RF"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % n_estimators )

if (len(n_list)==0):
    mae.append(1213)
    comb.append("RF" + " %s" % 50 )    
    
print(n_list.shape)
##Set figure size
plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
fig, ax = plt.subplots()
plt.plot(mae)
##Set the tick names to names of combinations
ax.set_xticks(range(len(comb)))
ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
plt.show()    

#Best estimated performance is 1213 when the number of estimators is 50


# ## Evaluation, prediction, and analysis
# * XGBoost

# In[ ]:


#Evaluation of various combinations of XGB

#Import the library
from xgboost import XGBRegressor

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([100, 200, 300, 500, 700, 1000, 1200, 1500])

for n_estimators in n_list:
    #Set the base model
    model = XGBRegressor(n_estimators=n_estimators,seed=seed)
    
    algo = "XGB"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % n_estimators )

if (len(n_list)==0):
    mae.append(1169)
    comb.append("XGB" + " %s" % 1000 )    
    The app will 
##Set figure size
plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
fig, ax = plt.subplots()
plt.plot(mae)
##Set the tick names to names of combinations
ax.set_xticks(range(len(comb)))
ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
plt.show()    

#Best estimated performance is 1169 with n=1000


# ## Evaluation, prediction, and analysis
# * MLP (Deep Learning)

# ## Make Predictions

# In[49]:


# Make predictions using XGB as it gave the best estimated performance        

X = numpy.concatenate((X_train,X_val),axis=0)
# del X_train
# del X_val
Y = numpy.concatenate((Y_train,Y_val),axis=0)
# del Y_train
# del Y_val

n_estimators = 1000

#Best model definition
best_model = XGBRegressor(n_estimators=n_estimators,seed=seed)
best_model.fit(X,Y)
# del X
# del Y
#Read test dataset
dataset_test = pandas.read_csv("input/pml_example_submission.csv")
#Drop unnecessary columns
ID = dataset_test['id']
dataset_test.drop('id',axis=1,inplace=True)

#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset_test.iloc[:,i])
    feature = feature.reshape(dataset_test.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

# del cats

#Concatenate encoded attributes with continuous attributes
X_test = numpy.concatenate((encoded_cats,dataset_test.iloc[:,split:].values),axis=1)

# del encoded_cats
# del dataset_test

#Make predictions using the best model
predictions = numpy.expm1(best_model.predict(X_test))
# del X_test
# Write submissions to output file in the correct format
with open("submission.csv", "w") as subfile:
    subfile.write("id,loss\n")
    for i, pred in enumerate(list(predictions)):
        subfile.write("%s,%s\n"%(ID[i],pred))

