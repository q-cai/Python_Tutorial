
# coding: utf-8

# # Pima Indians Diabetes Data Analysis Demo in Python
#
# - __Dataset__: This opensource dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. More information can be found in [this UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes).
#
# - __Goal__: The objective is to predict whether a patient has diabetes based on the information about diagnostic measurements. The target/reponse variable is a binary variable indicating whether a patient has diabetes or not. So it is a classification problem in machine learning.
#
# - __Tool__: Information about how to use [PySpark](https://spark.apache.org/docs/1.6.1/api/python/index.html) for machine learning, please refer to the [official document](https://spark.apache.org/docs/1.6.1/ml-guide.html). In this document, we will transform pyspark data frame to pandas, and use Python for data anaysis.
#
# Note: For pure PySpark analysis/modeling for Pima Indians Diabetes data, please refer to the other document: `demo-PIMA-PySpark.ipynb`.

# ## Table of Contents
#
# 1. [Getting the Data](#context1)<br>
# 2. [Data Cleaning and Exploration](#context2)<br>
#     2.1. [Tranforming to Pandas Datatype](#context21)<br>
#     2.2. [Data Lookup](#context22)<br>
#     2.3. [Data Visiualization](#context23)<br>
# 3. [Feature Engineering](#context3)<br>
# 4. [Machine Learning Modeling](#context4)<br>
#     4.1. [Train/Test Spliting](#context41)<br>
#     4.2. [Pupular Machine Learning Models](#context42)<br>
#     4.3. [Hyperparameters Tunning (KNN)](#context43)<br>
# 5. [Summary](#context5)<br>

# <a id="context1"></a>
# ## 1. Getting the Data
#
# Pima Indians Diabetes (PIMA) data is stored in the data mart. For more information about retrieving data from data mart, please refer to [Watson Platform for Health GxP Analytics System](https://www.ibm.com/support/knowledgecenter/SSSMS8/content/whac_dsg_t_read_datamart_python.html).
#
# For this example, we first import `SQLContext` to initialize the Spark context.

# In[1]:

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


# We then provide the data mart database connection information.

# In[2]:

# Defining data mart credentials
user = "fmuxbuhc"
password = "BR9ZhZorLolI"
jdbcURL = "jdbc:db2://datamart-ds-db-4.dev1.whclsf1.watson-health.net:50443/DM10"
table = "TEST10"
prop = {"user": user, "password": password,
        "driver": "com.ibm.db2.jcc.DB2Driver", "sslConnection": "true"}


# Finally we read in PIMA data from data mart with the variables defined above and name it df. df is in pyspark.sql.dataframe type.

# In[3]:

df = sqlContext.read.jdbc(url=jdbcURL, table=table, properties=prop)


# <a id="context2"></a>
# ## 2. Data Cleaning and Exploration
#
# <a id="context21"></a>
# ### 2.1. Tranforming to Pandas Datatype
#
# If possible, we can transform pyspark data frame to pandas, and use python for data anaysis. For pure PySpark data analysis for WBC data, please refer to the other document: `demo-PIMA-PySpark.ipynb`.
#
# We import `pandas` modole and use the function `toPandas()`. Now __df__ is in 'pandas.DataFrame' type.

# In[4]:

import numpy as np


# In[5]:

import pandas as pd
df = df.toPandas()
type(df)


# <a id="context22"></a>
# ### 2.2. Data Lookup
#
# Attributes Information:
#
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)
#
# First we print out the size (shape) of the data.

# In[6]:

print("There are %d rows and %d columns in the data." % (df.shape[0], df.shape[1]))


# Then we print out the first five rows of the data.

# In[7]:

df.head(5)


# Check if there is any missing data.

# In[8]:

# checking for missing values
df.isnull().any()


# There are 500 out of 768 patients have diabetes and the rest 268 do not have diabetes.

# In[9]:

df.groupby("OUTCOME").size()


# <a id="context23"></a>
# ### 2.3. Data Visualization

# In[10]:

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# In[11]:

get_ipython().magic(u'matplotlib inline')


# In order to learn the distributions, we show the histograms of all features as well as response.

# In[12]:

df.hist(figsize=(10, 8))


# To further check the distributions, especially outliers in features and response, we present boxplots.

# In[13]:

df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(10, 8))


# The correlation matrix (measures __linear__ association) is also presented.

# In[15]:

column_x = df.columns[0:len(df.columns) - 1]
corr = df[df.columns].corr()
print(corr)


# The scatterplot matrix is a comprehensive way to learn the distribution for each variable and the relationship among all variables.

# In[14]:

pd.scatter_matrix(df, alpha=0.3, figsize=(14, 8), diagonal='kde')


# <a id="context3"></a>
# ## 3. Feature Engineering

# For features `GLUCOSE`, `BLOODPRESSURE`, `SKINTHICKNESS`, `INSULIN` and `BMI`, the values should not be zero. However, there are zero values for these features. This indicates that they are actually __missing__ rather than 0. As a result, we need to replace these zeros with means.

# In[15]:

zero_fields = ['GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI']
df[zero_fields] = df[zero_fields].replace(0, np.nan)
df[zero_fields] = df[zero_fields].fillna(df.mean())


# The imputed feature summaries are shown below.

# In[16]:

df.describe().T


# The features are not in the same unit, in order to make future algorithms to coverge fast, we should do standardization. Here we use `RobustScaler`. More information about different Feature Scaling tools can be found in the [official document](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler.fit_transform) and [this blog](http://benalexkeen.com/feature-scaling-with-scikit-learn/).

# In[17]:

from sklearn.preprocessing import RobustScaler

y = df['OUTCOME']
X = df.drop('OUTCOME', axis=1)
X_scaled = pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)


# <a id="context4"></a>
# ## 4. Machine Learning Modeling
#
# <a id="context41"></a>
# ### 4.1. Train/Test Spliting
#
# To fairly evaluate the models, we should randomly split the dat into two parts. Here the 75% of data are randomly selected for training and the rest 25% the data is held for evaluation only.

# In[18]:

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=100, stratify=y)

print(X_train.shape, X_test.shape, y_train.size, y_test.size)


# <a id="context42"></a>
# ### 4.2. Pupular Machine Learning Models
#
# There are thousands of machine learning models or even more. For this demo, here we only apply the most pupular eight classification models:
#
# 1. Logistic Regression
# 2. Gaussian Naive Bayes
# 3. K Nearest Neighbors
# 4. Decision Tree
# 5. Random Forest
# 6. Adaboost
# 7. Gradient Boosting
#
# There are many evaluation criteria for a classifier. Here we compute [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve), [F1 score](https://en.wikipedia.org/wiki/F1_score) and the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix). The training time is also printed out for comparison.

# In[23]:

# load algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import f1_score

# helper functions


def train_clf(clf, X_train, y_train):

    return clf.fit(X_train, y_train)


def pred_clf(clf, features, target):

    y_pred = clf.predict(features)
    return f1_score(target.values, y_pred, pos_label=1), metrics.roc_auc_score(target.values, y_pred), metrics.confusion_matrix(target.values, y_pred), metrics.classification_report(target.values, y_pred, target_names=['1', '0'])


def train_predict(clf, X_train, y_train, X_test, y_test):

    train_clf(clf, X_train, y_train)

    print("F1 score for training set is: {:.4f}".format(pred_clf(clf, X_train, y_train)[0]))
    print("F1 score for testing set is: {:.4f}".format(pred_clf(clf, X_test, y_test)[0]))
    print("ROC AUC score for training set is: {:.4f}".format(pred_clf(clf, X_train, y_train)[1]))
    print("ROC AUC score for testing set is: {:.4f}".format(pred_clf(clf, X_test, y_test)[1]))
    print("Confusion Matrix on test set is : \n  {} ".format(pred_clf(clf, X_test, y_test)[2]))
    print("Classification summary report on test set is : \n  {} ".format(
        pred_clf(clf, X_test, y_test)[3]))


# In[24]:

# load algorithms
lrc = LogisticRegression()
nbc = GaussianNB()
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
abc = AdaBoostClassifier(random_state=0)
gbc = GradientBoostingClassifier(random_state=0)

algorithms = [lrc, nbc, knn, dtc, rfc, abc, gbc]

for clf in algorithms:
    """
    print("\n{}: \n".format(clf.__class__.__name__))

    # create training data from first 100, then 200, then 300
    #for n in [179, 358, 537]:
        #train_predict(clf, X_train[:n], y_train[:n], X_test, y_test)
    """
    print("{}:".format(clf))
    train_predict(clf, X_train, y_train, X_test, y_test)


# <a id="context43"></a>
# ### 4.3. Hyperparameters Tunning (KNN)

# In[25]:

# split training set into training and testing set
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
    X_train, y_train, test_size=0.3, random_state=100)
for n in range(3, 10):
    knn = KNeighborsClassifier(n_neighbors=n)
    print("Number of neighbors is: {}".format(n))
    train_predict(knn, X_train_cv, y_train_cv, X_test_cv, y_test_cv)


# In[26]:

from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=5)
clf_ = knn.fit(X_train, y_train)
y_pred = clf_.predict(X_test)
print('Accuracy is {}'.format(accuracy_score(y_test, y_pred)))


# In[27]:

knn


# <a id="context5"></a>
# ## 5. Summary
#
# In this demo, we explored the Wisconsin breast cancer data in Watson Health Platform 2.0.
# We retrieved data from data mart, then use pure Python for data exploration, cleaning and machine learning.

# ### Authors
# Quan Cai, data scientist, IBM Watson Platform for Health.
