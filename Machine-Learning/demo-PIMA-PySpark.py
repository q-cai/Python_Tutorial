
# coding: utf-8

# # Pima Indians Diabetes Data Analysis Demo in Python
# 
# - __Dataset__: This opensource dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. More information can be found in [this UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes).
# 
# - __Goal__: The objective is to predict whether a patient has diabetes based on the information about diagnostic measurements. The target/reponse variable is a binary variable indicating whether a patient has diabetes or not. So it is a classification problem in machine learning.
# 
# - __Tool__: Information about how to use [PySpark](https://spark.apache.org/docs/1.6.1/api/python/index.html) for machine learning, please refer to the [official document](https://spark.apache.org/docs/1.6.1/ml-guide.html).
# 
# Note: If possible, we can transform pyspark data frame to pandas, and use python for data anaysis. For pure Python data analysis for this data, please refer to the other document: `demo-PIMA-Python.ipynb`.

# ## Table of Contents
# 
# 1. [Getting the Data](#context1)<br>
# 2. [Data Cleaning and Exploration](#context2)<br>
#     2.1. [Data Lookup](#context21)<br>
#     2.2. [Data Type Change](#context22)<br>
#     2.3. [Label/Features Format Conversion](#context23)<br>
#     2.4. [Features Standardization](#context24)<br>
# 3. [Machine Learning Modeling](#context3)<br>
#     3.1. [Train/Test Spliting](#context31)<br>
#     3.2. [Applying and Evaluating Models](#context32)<br>
# 4. [Summary](#context4)<br>

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

#Defining data mart credentials 
user = "fmuxbuhc"
password = "BR9ZhZorLolI"
jdbcURL = "jdbc:db2://datamart-ds-db-4.dev1.whclsf1.watson-health.net:50443/DM10"
table = "TEST10"
prop = {"user":user, "password":password, "driver":"com.ibm.db2.jcc.DB2Driver", "sslConnection":"true"}


# Finally we read in PIMA data from data mart with the variables defined above and name it df. df is in pyspark.sql.dataframe type.

# In[3]:

df = sqlContext.read.jdbc(url=jdbcURL, table=table, properties=prop)


# In[4]:

type(df)


# <a id="context2"></a>
# ## 2. Data Cleaning and Exploration
# 
# <a id="context21"></a>
# ### 2.1. Data Lookup

# First we print out number of rows and number of columns of the data.

# In[5]:

print("There are %d rows and %d columns in the data." % (df.count(), len(df.columns)))


# Then we print the schema to know the more about the data type for each feature. 

# In[6]:

df.printSchema()


# Then we output the first five rows of PIMA data to have a glimpse.

# In[7]:

df.show(5)


# <a id="context22"></a>
# ### 2.2. Data Type Change

# The `long` type of numerical features is not suitable for further modeling, so we iterately change the type to `double`.

# In[8]:

for numCol in df.schema.names:
    df = df.withColumn(numCol, df[numCol].cast('double'))
df.printSchema()


# <a id="context23"></a>
# ### 2.3. Label/Features Format Conversion
# 
# In order for Spark to understand the data, we should convert features and target to label/feature format.

# In[9]:

from pyspark.sql import Row
from pyspark.mllib.linalg import DenseVector
from pyspark.ml.feature import OneHotEncoder, StringIndexer

labelCol = ['OUTCOME']

featuresCol = [c for c in df.columns if c not in {'OUTCOME'}]


row = Row('unscaledFeatures', 'unlabel')
 
# df = df[featuresCol + labelCol]

# 0-features, 1-label
# map features to DenseVector
lf = df.map(lambda r: (row(DenseVector(r[:-1]), r[-1]))).toDF()

# index label
# convert numeric label to categorical, which is required by
# decisionTree and randomForest
lf = StringIndexer(inputCol = 'unlabel', outputCol = 'label').fit(lf).transform(lf)

lf.show(3)


# Double check that the data version is correct after conversion.

# In[10]:

lf.take(2)


# <a id="context24"></a>
# ### 2.4. Features Standardization
# 
# The features are not in the same unit, in order to make future algorithms to coverge fast, we should do standardization. Here we standardize each feature to have zero mean and unit variance.

# In[11]:

from pyspark.ml.feature import StandardScaler

# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="unscaledFeatures", outputCol="features")

# Fit the DataFrame to the scaler
scaler = standardScaler.fit(lf)

# Transform the data in `df` with the scaler
scaled_df = scaler.transform(lf)

# Inspect the result
scaled_df.show(3)


# <a id="context3"></a>
# ## 3. Machine Learning Modeling
# 
# <a id="context31"></a>
# ### 3.1. Train/Test Spliting
# 
# To fairly evaluate the models, we should randomly split the dat into two parts. Here the 75% of data are randomly selected for training and the rest 25% the data is held for evaluation only.

# In[12]:

# random split further to get train/validate
(trainData,testData) = scaled_df.randomSplit([0.75,0.25], seed =111)
 
print('The number of training data: %d' % trainData.count())
print('The number of testing data: %d'  % testData.count())


# <a id="context32"></a>
# ### 3.2. Applying and Evaluating Models
# 
# There are hundreds of machine learning models or even more. For demo, here we only apply three most pupular classification models -- logistic regression, decision trees and random forest. For more information about classification in PySpark, please refer to the [offiical document](https://spark.apache.org/docs/1.6.1/api/python/pyspark.ml.html#module-pyspark.ml.classification).
# 
# There are many evaluation criteria for a classifier. Here we compute [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve), [F1 score](https://en.wikipedia.org/wiki/F1_score) and the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).
# 
# <a id="context321"></a>
# #### 3.2.1. Logistic Regression
# 
# Note that there are over 20 features, and they are [multi-correlated](https://en.wikipedia.org/wiki/Multicollinearity). When applying logistic regression, we should add regularization terms for variable selection to avoid overfitting. In PySpark, the [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) (L1 penalty) and [Elastic Net](https://en.wikipedia.org/wiki/Elastic_net_regularization) (weighted L1 and L2 penaltiess) regularization is available for this purpose.

# In[13]:

from pyspark.ml.classification import LogisticRegression
# Evaluate model based on auc ROC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# Evaluate model based on F1 socre
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Evaluate model based on confusion matrix
from pyspark.mllib.evaluation import MulticlassMetrics

# model on training data regPara: lasso regularisation parameter (L1)
lrModel = LogisticRegression(regParam = 0.2).fit(trainData)

# make prediction on test data
pred = lrModel.transform(testData)

pred.select('label', 'prediction').show()


evaluator1 = BinaryClassificationEvaluator(labelCol = 'label', metricName="areaUnderROC")
evaluator2 = MulticlassClassificationEvaluator(labelCol = 'label', metricName="f1")
metrics = MulticlassMetrics(pred.select('label', 'prediction').rdd.map(tuple))

print('AUC ROC of Logistic Regression model is %f' % evaluator1.evaluate(pred))
print('F1 score of Logistic Regression model is %f' % evaluator2.evaluate(pred))
metrics.confusionMatrix().toArray().transpose()


# <a id="context322"></a>
# #### 3.2.2. Decision Tree

# In[14]:

from pyspark.ml.classification import DecisionTreeClassifier

# model on training data maxDepth is the hyperparameter
dtModel = DecisionTreeClassifier(maxDepth = 3).fit(trainData)

# make prediction on test data
pred = dtModel.transform(testData)

pred.select('label', 'prediction').show()


evaluator1 = BinaryClassificationEvaluator(labelCol = 'label', metricName="areaUnderROC")
evaluator2 = MulticlassClassificationEvaluator(labelCol = 'label', metricName="f1")
metrics = MulticlassMetrics(pred.select('label', 'prediction').rdd.map(tuple))

print('AUC ROC of Decision Tree model is %f' % evaluator1.evaluate(pred))
print('F1 score of Decision Tree model is %f' % evaluator2.evaluate(pred))
metrics.confusionMatrix().toArray().transpose()


# <a id="context323"></a>
# #### 3.2.3. Random Forest

# In[15]:

from pyspark.ml.classification import RandomForestClassifier

# model on training data numTrees is the hyperparameter
rfModel = RandomForestClassifier(numTrees = 100).fit(trainData)

# make prediction on test data
pred = rfModel.transform(testData)

pred.select('label', 'prediction').show()


evaluator1 = BinaryClassificationEvaluator(labelCol = 'label', metricName="areaUnderROC")
evaluator2 = MulticlassClassificationEvaluator(labelCol = 'label', metricName="f1")
metrics = MulticlassMetrics(pred.select('label', 'prediction').rdd.map(tuple))

print('AUC ROC of Random Forest model is %f' % evaluator1.evaluate(pred))
print('F1 score of Random Forest model is %f' % evaluator2.evaluate(pred))
metrics.confusionMatrix().toArray().transpose()


# <a id="context4"></a>
# ## 4. Summary
# 
# In this demo, we explored the PIMA Indians Diabetes dataset in Watson Health Platform 2.0.
# We retrieved data from data mart, then use PySpark for data exploration, cleaning and machine learning. 
# For this classification problem, we have tried logistic regression, decision trees and random forest models and evaluate the prediction accuracy on AUC, F1 score and confusion matrix without any hyperparameter tuning.

# ### Authors
# Quan Cai, data scientist, IBM Watson Platform for Health.
# 
# Date: 11/07/2017
