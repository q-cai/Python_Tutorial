
# coding: utf-8

# # Chronic Kidney Disease Data Analysis Demo in PySpark
#
# - __Dataset__: This opensource dataset can be used to predict the chronic kidney disease and it can be collected from the hospital nearly 2 months of period. More information can be found in [this UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease#).
#
# - __Goal__: The target/reponse variable is a binary variable indicating whether a patient has chronic kidney disease or not. So it is a classification problem in machine learning.
#
# - __Tool__: Information about how to use [PySpark](https://spark.apache.org/docs/1.6.1/api/python/index.html) for machine learning, please refer to the [official document](https://spark.apache.org/docs/1.6.1/ml-guide.html).
#
# Note: If possible, we can transform pyspark data frame to pandas, and use python for data anaysis. For pure Python data analysis for CKD data, please refer to the other document: `demo-CKD-Python.ipynb`.

# ## Table of Contents
#
# 1. [Getting the Data](#context1)<br>
# 2. [Data Cleaning and Exploration](#context2)<br>
#     2.1. [Data Lookup](#context21)<br>
#     2.2. [Numerical Feature Engineering](#context22)<br>
#     2.3. [Categorical Feature Engineering](#context23)<br>
#     2.4. [Label/Features Format Conversion](#context24)<br>
#     2.5. [Features Standardization](#context25)<br>
# 3. [Machine Learning Modeling](#context3)<br>
#     3.1. [Train/Test Spliting](#context31)<br>
#     3.2. [Applying and Evaluating Models](#context32)<br>
# 4. [Summary](#context4)<br>

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

user = ""
password = ""
jdbcURL = ""
table = "TEST8"
prop = {"user": user, "password": password,
        "driver": "com.ibm.db2.jcc.DB2Driver", "sslConnection": "true"}


# Finally we read in CKD data from data mart with the variables defined above and name it __df__.

# In[3]:

df = sqlContext.read.jdbc(url=jdbcURL, table=table, properties=prop)


# __df__ is in `pyspark.sql.dataframe` type.

# In[4]:

type(df)


# <a id="context2"></a>
# ## 2. Data Cleaning and Exploration

# <a id="context21"></a>
# ### 2.1. Data Lookup

# In[5]:

print("There are %d rows and %d columns in the data." % (df.count(), len(df.columns)))


# First we print the schema to know the more about the data type for each feature. Observing that most features are in `string` type, AGE and BP are in `long` type, we will need to change types of features later on ([Section 2.2](#context22) and [Section 2.3](#context23)).

# In[6]:

df.printSchema()


# Then we output the first five rows of CKD data. We can see that there are missing values, indiciated by "?" sign. Therefore, we will need to impute missing values later on.

# In[7]:

df.show(5)


# Features are in either numerical or categorical type. We separate them for later analysis. The target/response is organized individually.

# In[8]:

numVars = ['AGE', 'BP', 'SG', 'AL', 'SU', 'BGR', 'BU',
           'SC', 'SOD', 'POT', 'HEMO', 'PCV', 'WBCC', 'RBCC']
catVars = ['RBC', 'PC', 'PCC', 'BA', 'HTN', 'DM', 'CAD', 'APPET', 'PE', 'ANE']
labelCol = ['CLASS']


# <a id="context22"></a>
# ### 2.2. Numerical Feature Engineering
#
# <a id="context221"></a>
# #### 2.2.1. Data Type Change

# In[9]:

for numCol in numVars:
    df = df.withColumn(numCol, df[numCol].cast('double'))
df.printSchema()


# <a id="context222"></a>
# #### 2.2.2. Imputation With Mean

# In[10]:

from pyspark.sql.functions import avg
imputeDF = df
for c in numVars:
    meanValue = imputeDF.agg(avg(c)).first()[0]
    print(c, meanValue)
    imputeDF = imputeDF.na.fill(meanValue, [c])


# We output the imputed data to make sure that the missing values for all numerical features are indeed imputed.

# In[11]:

imputeDF.show(5)


# <a id="context23"></a>
# ### 2.3. Categorical Feature Engineering
#
# <a id="context231"></a>
# #### 2.3.1. Category Indexing
#
# The categorical features should be transformed into numerical ones. The most popular ways for transformation is by `StringIndexer` or `OneHotEncoder`. For simplicity, here we apply `StringIndexer`.

# In[12]:

from pyspark.ml.feature import OneHotEncoder, StringIndexer

# make use of pipeline to index all categorical variables
# the indexed categorical features are stored with name 'original colname' + 'indexed'


def indexer(df, col):
    si = StringIndexer(inputCol=col, outputCol=col + '_indexed').fit(df)
    return si


indexers = [indexer(imputeDF, col) for col in catVars]

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=indexers)
df_indexed = pipeline.fit(imputeDF).transform(imputeDF)

df_indexed.show(3)


# <a id="context24"></a>
# ### 2.4. Label/Features Format Conversion
#
# In order for Spark to understand the data, we should convert features and target to label/feature format. Make sure that we only select the features we want, which are the imputed numerical features and the indexed categorical features.

# In[13]:

catVarsIndexed = [i + '_indexed' for i in catVars]
featuresCol = numVars + catVarsIndexed

from pyspark.sql import Row
from pyspark.mllib.linalg import DenseVector
row = Row('catLabel', 'unscaledFeatures')

df_indexed = df_indexed[labelCol + featuresCol]

# 0-label, 1-features
# map features to DenseVector
lf = df_indexed.map(lambda r: (row(r[0], DenseVector(r[1:])))).toDF()

# index label
# convert numeric label to categorical, which is required by
# decisionTree and randomForest
lf = StringIndexer(inputCol='catLabel', outputCol='label').fit(lf).transform(lf)

lf.show(3)


# <a id="context25"></a>
# ### 2.5. Features Standardization
#
# The features are not in the same unit, in order to make future algorithms to coverge fast, we should do standardization. Here we standardize each feature to have zero mean and unit variance.

# In[14]:

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

# In[15]:

# random split further to get train/validate
(trainData, testData) = scaled_df.randomSplit([0.75, 0.25], seed=111)

print('The number of training data: %d' % trainData.count())
print('The number of testing data: %d' % testData.count())


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

# In[17]:

from pyspark.ml.classification import LogisticRegression
# Evaluate model based on auc ROC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# Evaluate model based on F1 socre
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Evaluate model based on confusion matrix
from pyspark.mllib.evaluation import MulticlassMetrics

# model on training data regPara: lasso regularisation parameter (L1)
lrModel = LogisticRegression(regParam=0.1).fit(trainData)

# make prediction on test data
pred = lrModel.transform(testData)

pred.select('catLabel', 'label', 'prediction').show()


evaluator1 = BinaryClassificationEvaluator(labelCol='label', metricName="areaUnderROC")
evaluator2 = MulticlassClassificationEvaluator(labelCol='label', metricName="f1")
metrics = MulticlassMetrics(pred.select('label', 'prediction').rdd.map(tuple))

print('AUC ROC of Logistic Regression model is %f' % evaluator1.evaluate(pred))
print('F1 score of Logistic Regression model is %f' % evaluator2.evaluate(pred))
metrics.confusionMatrix().toArray().transpose()


# <a id="context322"></a>
# #### 3.2.2. Decision Tree

# In[18]:

from pyspark.ml.classification import DecisionTreeClassifier

# model on training data maxDepth is the hyperparameter
dtModel = DecisionTreeClassifier(maxDepth=3).fit(trainData)

# make prediction on test data
pred = dtModel.transform(testData)

pred.select('catLabel', 'label', 'prediction').show()


evaluator1 = BinaryClassificationEvaluator(labelCol='label', metricName="areaUnderROC")
evaluator2 = MulticlassClassificationEvaluator(labelCol='label', metricName="f1")
metrics = MulticlassMetrics(pred.select('label', 'prediction').rdd.map(tuple))

print('AUC ROC of Decision Tree model is %f' % evaluator1.evaluate(pred))
print('F1 score of Decision Tree model is %f' % evaluator2.evaluate(pred))
metrics.confusionMatrix().toArray().transpose()


# <a id="context323"></a>
# #### 3.2.3. Random Forest

# In[19]:

from pyspark.ml.classification import RandomForestClassifier

# model on training data numTrees is the hyperparameter
rfModel = RandomForestClassifier(numTrees=100).fit(trainData)

# make prediction on test data
pred = rfModel.transform(testData)

pred.select('catLabel', 'label', 'prediction').show()


evaluator1 = BinaryClassificationEvaluator(labelCol='label', metricName="areaUnderROC")
evaluator2 = MulticlassClassificationEvaluator(labelCol='label', metricName="f1")
metrics = MulticlassMetrics(pred.select('label', 'prediction').rdd.map(tuple))

print('AUC ROC of Random Forest model is %f' % evaluator1.evaluate(pred))
print('F1 score of Random Forest model is %f' % evaluator2.evaluate(pred))
metrics.confusionMatrix().toArray().transpose()


# <a id="context4"></a>
# ## 4. Summary
#
# In this demo, we explored the chronic disease dataset in Watson Health Platform 2.0.
# We retrieved data from data mart, then use PySpark for data exploration, cleaning and machine learning.
# For this classification problem, there are both categorical and numerical values missing. We impute them with mean for numerical features and treat missing values as a new category for categorical features.
# With the clean data, logistic regression, decision trees and random forest models all have very high prediction accuracy on AUC, F1 score and confusion matrix without any hyperparameter tuning.

# ### Authors
# Quan Cai, data scientist, IBM Watson Platform for Health.
