
# coding: utf-8

# # Wisconsin Breast Cancer Data Analysis Demo in Python
#
# - __Dataset__: This opensource dataset can be used to classify breast tumors as malignant or benign depending on measurements of the tumor cells. More information can be found in <a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29" target="_blank" rel="noopener noreferrer">this UCI Machine Learning Repository</a>.
#
# - __Goal__: The target/reponse variable is a binary variable indicating whether a breast tumor is malignant or benign. So it is a classification problem in machine learning. We would like to develop a predictive model which classifies breast tumors as malignant or benign depending on measurements of the tumor cells.
#
# - __Tool__: Information about how to use [PySpark](https://spark.apache.org/docs/1.6.1/api/python/index.html) for machine learning, please refer to the [official document](https://spark.apache.org/docs/1.6.1/ml-guide.html). In this document, we will transform pyspark data frame to pandas, and use Python for data anaysis.
#
# Note: For pure PySpark analysis/modeling for Wisconsin breast cancer data, please refer to the other document: `demo-WBC-PySpark.ipynb`.

# ## Table of Contents
#
# 1. [Getting the Data](#context1)<br>
# 2. [Data Cleaning and Exploration](#context2)<br>
#     2.1. [Tranforming to Pandas Datatype](#context21)<br>
#     2.2. [Data Loopup](#context22)<br>
# 3. [EDA: Scatterplot Matrix](#context3)<br>
# 4. [Feature Engineering](#context4)<br>
#     4.1. [Missing Data Imputation](#context41)<br>
#     4.2. [Category Indexing](#context42)<br>
#     4.3. [Scaling Features](#context43)<br>
#     4.4. [Dimension Reduction](#context44)<br>
# 5. [Machine Learning Modeling](#context5)<br>
#     5.1. [Train/Test Spliting](#context51)<br>
#     5.2. [Applying and Evaluating Models](#context52)<br>
# 6. [Summary](#context6)<br>

# <a id="context1"></a>
# ## 1. Getting the Data
#
# Wisconsin breast cancer (WBC) data is stored in the data mart. For more information about retrieving data from data mart, please refer to [Watson Platform for Health GxP Analytics System](https://www.ibm.com/support/knowledgecenter/SSSMS8/content/whac_dsg_t_read_datamart_python.html).
#
# For this example, we first import `SQLContext` to initialize the Spark context.

# In[1]:

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


# We then provide the data mart database connection information.

# In[2]:

user = "dvnbkwjk"
password = "bIj1aX39n8eL"
jdbcURL = "jdbc:db2://datamart-ds-db-2.dev1.whclsf1.watson-health.net:50443/DM6"
table = "TEST9"
prop = {"user": user, "password": password,
        "driver": "com.ibm.db2.jcc.DB2Driver", "sslConnection": "true"}


# Finally we read in WBC data from data mart with the variables defined above and name it df.

# In[3]:

df = sqlContext.read.jdbc(url=jdbcURL, table=table, properties=prop)


# df is in `pyspark.sql.dataframe` type.

# In[4]:

type(df)


# <a id="context2"></a>
# ## 2. Data Cleaning and Exploration
#
# <a id="context21"></a>
# ### 2.1. Tranforming to Pandas Datatype
#
# If possible, we can transform pyspark data frame to pandas, and use python for data anaysis. For pure PySpark data analysis for WBC data, please refer to the other document: `demo-WBC-PySpark.ipynb`.
#
# We import `pandas` modole and use the function `toPandas()`. Now __df__ is in 'pandas.DataFrame' type.

# In[5]:

import pandas as pd
df = df.toPandas()
type(df)


# <a id="context22"></a>
# ### 2.2. Data Lookup

# The data contains features extracted from 569 diagnostic images of breast tumors. The diagnosis column indicates whether the mass was benign (B), or malignant (M). The rest of the columns contain features which are structured as follows:
#
# 10 variables describe the cell nuclei of each mass, and for each variable, the mean, standard deviation, and 'worst' (mean of three largest measurements) are calculated.

# In[6]:

print("There are %d rows and %d columns in the data." % (df.shape[0], df.shape[1]))


# We output the first five rows of WBC data.

# In[7]:

df.head(5)


# Check if there is any missing data.

# In[8]:

# checking for missing values
df.isnull().any()


# Converting diagnosis M/B to numerical values 0/1.

# In[9]:

from sklearn.preprocessing import LabelEncoder

# converting diagnosis M/B to numerical
lenc = LabelEncoder()
lenc.fit(df['DIAGNOSIS'])
df['DIAGNOSIS'] = lenc.transform(df['DIAGNOSIS'])

df.head()


# In[10]:

df = df.drop(["ID"], axis=1)
df.head()


# In[11]:

df.shape


# <a id="context3"></a>
# ## 3. Scaling Features
#
# The features are not in the same unit, in order to make future algorithms to coverge fast, we should do standardization. Here we use `RobustScaler`. More information about different Feature Scaling tools can be found in the [official document](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler.fit_transform) and [this blog](http://benalexkeen.com/feature-scaling-with-scikit-learn/).

# In[12]:

from sklearn.preprocessing import RobustScaler
y = df['DIAGNOSIS']
X = df.drop('DIAGNOSIS', axis=1)
X_scaled = pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)


# <a id="context4"></a>
# ## 4. Machine Learning Modeling
#
# <a id="context41"></a>
# ### 4.1. Train/Test Spliting
#
# To fairly evaluate the models, we should randomly split the dat into two parts. Here the 80% of data are randomly selected for training and the rest 20% the data is held for evaluation only.

# In[13]:

from sklearn.cross_validation import train_test_split

# X, y = df.iloc[:,1:], df[['DIAGNOSIS']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y)


# In[14]:

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# <a id="context42"></a>
# ### 4.2. Decision Tree Model

# In[15]:

from sklearn import tree

# initializing the tree model and training it
tree_model = tree.DecisionTreeClassifier()
tree_model = tree_model.fit(X_train, y_train)


# In[16]:

from sklearn import metrics

# Predict test set:
y_pred = tree_model.predict(X_test)
# test_predprob = tree_model.predict_proba(X_test)[:,1]

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['Benign', 'Malignant']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))


# <a id="context43"></a>
# ### 4.3. Logistic Regression Model

# In[17]:

from sklearn.linear_model import LogisticRegression

# initializing the tree model and training it
lr_model = LogisticRegression()
lr_model = lr_model.fit(X_train, y_train)

# Predict test set:
y_pred = lr_model.predict(X_test)

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['Benign', 'Malignant']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))


# <a id="context44"></a>
# ### 4.4. Naive Bayes Model

# In[18]:

from sklearn.naive_bayes import GaussianNB

# initializing the tree model and training it
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

# Predict test set:
y_pred = nb_model.predict(X_test)

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['Benign', 'Malignant']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))


# <a id="context45"></a>
# ### 4.5. Support Vector Machine Model

# In[19]:

from sklearn.svm import SVC

# initializing the tree model and training it
sv_model = SVC()
sv_model = sv_model.fit(X_train, y_train)

# Predict test set:
y_pred = sv_model.predict(X_test)

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['Benign', 'Malignant']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))


# <a id="context46"></a>
# ### 4.6. Random Forest Model

# In[20]:

from sklearn.ensemble import RandomForestClassifier

# initializing the tree model and training it
rf_model = RandomForestClassifier()
rf_model = rf_model.fit(X_train, y_train)

# Predict test set:
y_pred = rf_model.predict(X_test)

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['Benign', 'Malignant']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))


# <a id="context47"></a>
# ### 4.7. K Nearest Neighbors Model

# In[21]:

from sklearn.neighbors import KNeighborsClassifier

# initializing the tree model and training it
kn_model = KNeighborsClassifier()
kn_model = kn_model.fit(X_train, y_train)

# Predict test set:
y_pred = kn_model.predict(X_test)

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['Benign', 'Malignant']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))


# <a id="context5"></a>
# ## 5. Summary
#
# In this demo, we explored the Wisconsin breast cancer data in Watson Health Platform 2.0.
# We retrieved data from data mart, then use pure Python for data exploration, cleaning and machine learning.

# In[ ]:
