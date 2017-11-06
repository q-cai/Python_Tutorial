
# coding: utf-8

# # Chronic Kidney Disease Data Analysis Demo in Python
#
# - __Dataset__: This opensource dataset can be used to predict the chronic kidney disease and it can be collected from the hospital nearly 2 months of period. More information can be found in [this UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease#).
#
# - __Goal__: The target/reponse variable is a binary variable indicating whether a patient has chronic kidney disease or not. So it is a classification problem in machine learning.
#
# - __Tool__: Information about how to use [PySpark](https://spark.apache.org/docs/1.6.1/api/python/index.html) for machine learning, please refer to the [official document](https://spark.apache.org/docs/1.6.1/ml-guide.html). In this document, we will transform pyspark data frame to pandas, and use Python for data anaysis.
#
# Note: For pure PySpark analysis/modeling for chronic kidney disease data, please refer to the other document: `demo-CKD-PySpark.ipynb`.

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
# The chronic kidney disease (CKD) data is stored in the data mart. For more information about retrieving data from data mart, please refer to [Watson Platform for Health GxP Analytics System](https://www.ibm.com/support/knowledgecenter/SSSMS8/content/whac_dsg_t_read_datamart_python.html).
#
# For this example, we first import `SQLContext` to initialize the Spark context.

# In[1]:

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


# We then provide the data mart database connection information.

# In[2]:

# Defining data mart credentials
user = "rebtveec"
password = "A2jtIvJnDEya"
jdbcURL = "jdbc:db2://datamart-ds-db-1.dev1.whclsf1.watson-health.net:50443/DM3"
table = "TEST8"
prop = {"user": user, "password": password,
        "driver": "com.ibm.db2.jcc.DB2Driver", "sslConnection": "true"}


# Finally we read in CKD data from data mart with the variables defined above and the retrived data is in pyspark.sql.dataframe format.

# In[3]:

df0 = sqlContext.read.jdbc(url=jdbcURL, table=table, properties=prop)


# <a id="context2"></a>
# ## 2. Data Cleaning and Exploration
#
# <a id="context21"></a>
# ### 2.1. Tranforming to Pandas Datatype
#
# If possible, we can transform pyspark data frame to pandas, and use python for data anaysis. For pure PySpark data analysis for CKD data, please refer to the other document: `demo-CKD-PySpark.ipynb`.
#
# We import `pandas` modole and use the function `toPandas()`. Now __df__ is in 'pandas.DataFrame' type.

# In[4]:

import pandas as pd
df = df0.toPandas()
type(df)


# In[5]:

# show all available pakcages
# ! pip list


# <a id="context22"></a>
# ### 2.2. Data Lookup

# In[6]:

print("There are %d rows and %d columns in the data." % (df.shape[0], df.shape[1]))


# We output the first five rows of CKD data. We can see that there are missing values, indiciated by "?" sign.

# In[7]:

df.head(5)


# We replace missing values with "?" signs with NA that Python understand.

# In[8]:

import numpy as np
df = df.replace('?', np.NaN)
df.head(5)


# Then we print the data type for each feature. Observing that most features are in `object` type, AGE and BP are in `int` type, we will need to change types of features.

# In[9]:

df.dtypes


# Features are in either numerical or categorical type. We separate them and change all numerical features into `float` types.

# In[10]:

numVars = ['AGE', 'BP', 'SG', 'AL', 'SU', 'BGR', 'BU',
           'SC', 'SOD', 'POT', 'HEMO', 'PCV', 'WBCC', 'RBCC']
catVars = ['RBC', 'PC', 'PCC', 'BA', 'HTN', 'DM', 'CAD', 'APPET', 'PE', 'ANE', 'CLASS']

for col in numVars:
    df[col] = df[col].astype(float)


# In[11]:

df.head(5)


# In[12]:

n_ckd = len(df[df['CLASS'] == 'ckd'])
n_nockd = len(df[df['CLASS'] == 'notckd'])
print("%d people have chronic kidney disease," % n_ckd)
print("%d people don't have chronic kidney disease." % n_nockd)


# <a id="context3"></a>
# ## 3. EDA: Scatterplot Matrix

# In[13]:

get_ipython().magic(u'matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix
scatter_matrix(df, alpha=0.6, figsize=(10, 10), diagonal='kde')
plt.show()


# <a id="context4"></a>
# ## 4. Feature Engineering
#
# <a id="context41"></a>
# ### 4.1. Missing Data Imputation
#
# We fill in missing values with most frequent for nominal and median for numerical features.

# In[14]:

X = pd.DataFrame(df)
fill = pd.Series([X[c].value_counts().index[0]
                  if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                 index=X.columns)
new_ckd = X.fillna(fill)
print(new_ckd.head(10))


# <a id="context42"></a>
# ### 4.2. Category Indexing
#
# Then we nominalize categorical features. Label Encoding on the nominal values. More information can be found in the [official document](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). We may use [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) as well.

# In[15]:

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ckd2 = new_ckd.copy()
for items in ckd2:
    if ckd2[items].dtype == np.dtype('O'):
        ckd2[items] = le.fit_transform(ckd2[items])

print(ckd2.dtypes)
print(ckd2.head(10))


# <a id="context43"></a>
# ### 4.3. Scaling Features
#
# The features are not in the same unit, in order to make future algorithms to coverge fast, we should do standardization. Here we use `RobustScaler`. More information about different Feature Scaling tools can be found in the [official document](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler.fit_transform) and [this blog](http://benalexkeen.com/feature-scaling-with-scikit-learn/).

# In[16]:

from sklearn.preprocessing import RobustScaler
target_class = ckd2['CLASS']
features = ckd2.drop('CLASS', axis=1)
ckd2_robust = pd.DataFrame(RobustScaler().fit_transform(features), columns=features.columns)


# <a id="context44"></a>
# ### 4.4. Dimensionality Reduction
#
# Since they are over 20 features, and from the exploratory data analysis, some features are highly correlated. We need to do feature selection and dimension reduction. Here we apply the linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. More information about different Feature Scaling tools can be found in the [official document](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA).

# In[17]:

from sklearn.decomposition import PCA
pca = PCA()
ckd2_pca = pd.DataFrame(pca.fit_transform(ckd2_robust), columns=ckd2_robust.columns)
print(pca.explained_variance_ratio_.cumsum())


# First 11 components gives principal components that explain more than 95% of total variance. So let the number of components be 11.

# In[18]:

pca = PCA(n_components=11)
pca.fit(ckd2_robust)
reduced_ckd2 = pca.transform(ckd2_robust)
reduced_ckd2 = pd.DataFrame(reduced_ckd2, columns=[
                            'dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6', 'dim7', 'dim8', 'dim9', 'dim10', 'dim11'])
print(reduced_ckd2.head(10))


# <a id="context5"></a>
# ## 5. Machine Learning Modeling
#
# <a id="context51"></a>
# ### 5.1. Train/Test Spliting
#
# To fairly evaluate the models, we should randomly split the dat into two parts. Here the 75% of data are randomly selected for training and the rest 25% the data is held for evaluation only.

# In[19]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    reduced_ckd2, target_class, test_size=0.25, random_state=123)


# <a id="context52"></a>
# ### 5.2 Applying and Evaluating Models
#
# There are hundreds of machine learning models or even more. For demo, here we only apply the most pupular six classification models:
#
# 1. Gaussian Naive Bayes
# 2. Decision Tree
# 3. Support Vector Classification
# 4. Random Forest
# 5. K Nearest Neighbors
# 6. Logistic Regression
#
# There are many evaluation criteria for a classifier. Here we compute [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve), [F1 score](https://en.wikipedia.org/wiki/F1_score) and the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix). The training time is also printed out for comparison.

# In[20]:

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

from time import time

clf_1 = GaussianNB()
clf_2 = tree.DecisionTreeClassifier(random_state=123)
clf_3 = SVC()
clf_4 = RandomForestClassifier()
clf_5 = KNeighborsClassifier()
clf_6 = LogisticRegression()

# print the training timer for each classifier
for clf in [clf_1, clf_2, clf_3, clf_4, clf_5, clf_6]:
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("\n{}: \n".format(clf.__class__.__name__) +
          " trained model in {:.4f} milliseconds".format((end - start) * 1000))

target_names = ['CKD', 'Non-CKD']
# evaluate the performance of each classifier with F1 and AUC
for clf in [clf_1, clf_2, clf_3, clf_4, clf_5, clf_6]:
    y_pred = clf.predict(X_test)

    f1_scorer = metrics.f1_score(y_test, y_pred, pos_label=0)
    auc_scorer = metrics.roc_auc_score(y_test, y_pred)

    print("\n{}: \n".format(clf.__class__.__name__))
    print("F1 score for test test is {}".format(round(f1_scorer, 3)))
    print("AUC score for test test is {}".format(round(auc_scorer, 3)))

    print(" ")
    print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))

    print(" ")

    print(metrics.classification_report(y_test, y_pred, target_names=target_names))


# <a id="context6"></a>
# ## 6. Summary
#
# In this demo, we explored the chronic disease dataset in Watson Health Platform 2.0.
# We retrieved data from data mart, then use pure Python for data exploration, cleaning and machine learning.
# For this classification problem, there are both categorical and numerical values missing. We impute them with median for numerical features and treat missing values as a new category for categorical features.
# With the clean data, logistic regression, decision trees and random forest models all have very high prediction accuracy on AUC, F1 score and confusion matrix without any hyperparameter tuning.

# ### Authors
# Quan Cai, data scientist, IBM Watson Platform for Health.
