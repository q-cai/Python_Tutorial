
# coding: utf-8

# In[ ]:

# Invoke the SparkContext
sc


# In[ ]:

# Check the Spark version
sc.version


# <a id="context1"></a>
# ## 1. Retrieving Data From Data Mart
#
# The chronic kidney disease (CKD) data is stored in the data mart. For more information about retrieving data from data mart, please refer to [Watson Platform for Health GxP Analytics System](https://www.ibm.com/support/knowledgecenter/SSSMS8/content/whac_dsg_t_read_datamart_python.html).
#
# For this example, we first import `SQLContext` to initialize the Spark context.

# In[1]:

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


# In[2]:

# Defining data mart credentials
user = "rebtveec"
password = "A2jtIvJnDEya"
jdbcURL = "jdbc:db2://datamart-ds-db-1.dev1.whclsf1.watson-health.net:50443/DM3"
table = "TEST8"
prop = {"user": user, "password": password,
        "driver": "com.ibm.db2.jcc.DB2Driver", "sslConnection": "true"}


# In[3]:

df = sqlContext.read.jdbc(url=jdbcURL, table=table, properties=prop)


# <a id="context2"></a>
# ## 2. Data Lookup

# In[4]:

df.columns


# In[5]:

df.printSchema()


# In[6]:

print("There are %d rows and %d columns in the data." % (df.count(), len(df.columns)))


# In[7]:

df.first()


# In[8]:

df.take(2)


# In[9]:

df.show(2)


# In[10]:

df.describe().toPandas().transpose()


# In[11]:

numVars = ['AGE', 'BP', 'SG', 'AL', 'SU', 'BGR', 'BU',
           'SC', 'SOD', 'POT', 'HEMO', 'PCV', 'WBCC', 'RBCC']
catVars = ['RBC', 'PC', 'PCC', 'BA', 'HTN', 'DM', 'CAD', 'APPET', 'PE', 'ANE']
labelCol = ['CLASS']


# <a id="context3"></a>
# ## 3. Numerical Feature Engineering
#
# <a id="context31"></a>
# ### 3.1. Data Type Change

# In[12]:

for numCol in numVars:
    df = df.withColumn(numCol, df[numCol].cast('double'))


# In[13]:

df.describe().toPandas().transpose()


# In[14]:

df.stat.corr("PCV", "HEMO")


# In[15]:

from pyspark.sql.functions import *

df.agg(avg('AGE')).show()


# <a id="context32"></a>
# ### 3.2. Imputation With Mean
#
# Then we impute the mean value for each numerical feature. The mean for each numerical feature is printed for reference.

# In[16]:

for c in numVars:
    meanValue = df.agg(avg(c)).first()[0]
    print(c, meanValue)
    df = df.na.fill(meanValue, [c])


# <a id="context33"></a>
# ### 3.3. SQL Function Applications

# In[17]:

df.select("AGE", "CLASS").show(5)


# In[18]:

df.filter((df["AGE"] > 50) & (df["CLASS"] == 'ckd')).select("AGE", "CLASS").show(5)


# In[19]:

df.groupBy("CLASS").count().show()


# In[20]:

df.groupBy("CLASS").agg(avg("AGE")).show()


# In[21]:

df.select(mean('AGE'), sum('AGE')).show()


# <a id="context4"></a>
# ## 4. Work with Resilient Distributed Datasets
#
# Apache Spark uses an abstraction for working with data called a Resilient Distributed Dataset (RDD). An RDD is a collection of elements that can be operated on in parallel. RDDs are immutable, so you can't update the data in them. To update data in an RDD, you must create a new RDD. In Apache Spark, all work is done by creating new RDDs, transforming existing RDDs, or using RDDs to compute results. When working with RDDs, the Spark driver application automatically distributes the work across the cluster.
#
# You can construct RDDs by parallelizing existing Python collections (lists), by manipulating RDDs, or by manipulating files in HDFS or any other storage system.
#
# You can run these types of methods on RDDs:
#
# - Actions: query the data and return values
# - Transformations: manipulate data values and return pointers to new RDDs.
#
# Find more information on Python methods in the [PySpark documentation](http://spark.apache.org/docs/latest/api/python/pyspark.html).

# In[22]:

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# In[23]:

# Create an RDD
x_nbr_rdd = sc.parallelize(x)


# In[24]:

# View the first element in the RDD
x_nbr_rdd.first()


# In[25]:

# view the first five elements in the RDD
x_nbr_rdd.take(5)


# Run the `map()` function with the `lambda` keyword to replace each element, X, in your first RDD (the one that has numeric values) with X+1. Because RDDs are immutable, you need to specify a new RDD name.

# In[26]:

x_nbr_rdd_2 = x_nbr_rdd.map(lambda x: x + 1)
x_nbr_rdd_2.collect()


# Be careful with the `collect` method! It returns __all__ elements of the RDD to the driver. Returning a large data set might be not be very useful. No-one wants to scroll through a million rows!

# An array of values is a common data format where multiple values are contained in one element. You can manipulate the individual values if you split them up into separate elements.
#
# Create an array of numbers by including quotation marks around the whole set of numbers. If you omit the quotation marks, you get a collection of numbers instead of an array.

# In[27]:

X = ["1,2,3,4,5,6,7,8,9,10"]
y_rd = sc.parallelize(X)
Sum_rd = y_rd.map(lambda y: y.split(",")).map(lambda y: (int(y[2]) * int(y[8])))
Sum_rd.collect()


# Split and count text strings

# In[28]:

Words = ["Hello Human. I'm Apache Spark and I love running analysis on data."]
words_rd = sc.parallelize(Words)
words_rd.first()


# In[29]:

Words_rd2 = words_rd.map(lambda line: line.split(" "))
Words_rd2.first()


# In[30]:

Words_rd2.count()


# In[31]:

words_rd2 = words_rd.flatMap(lambda line: line.split(" "))
words_rd2.take(5)


# In[32]:

z = ["First,Line", "Second,Line", "and,Third,Line"]
z_str_rdd = sc.parallelize(z)
z_str_rdd.first()


# In[33]:

z_str_rdd_split_flatmap = z_str_rdd.flatMap(lambda line: line.split(","))
z_str_rdd_split_flatmap.collect()


# In[34]:

countWords = z_str_rdd_split_flatmap.map(lambda word: (word, 1))
countWords.collect()


# In[35]:

from operator import add
countWords2 = countWords.reduceByKey(add)
countWords2.collect()


# Filter data

# In[36]:

words_rd3 = z_str_rdd_split_flatmap.filter(lambda line: "Line" in line)

print "The count of words " + str(words_rd3.first())
print "Is: " + str(words_rd3.count())


# <a id="context5"></a>
# ## 5. Encoding and Assumbling Categorical Features
#
# From https://stackoverflow.com/questions/32982425/encode-and-assemble-multiple-features-in-pyspark

# ### Example 1

# In[37]:

from pyspark.sql import Row
from pyspark.mllib.linalg import DenseVector

row = Row('Gender', 'Age', 'Occupation', 'City_Category', 'Marital_Status')

df = sc.parallelize([
    row("M", 23, 'Farmer', 'Boston', 'U'),
    row("F", 35, 'Scientist', 'Boston', 'M'),
    row("F", 18, 'Designer', 'NYC', 'U'),
    row("M", 44, 'Designer', 'Dallas', 'M')
]).toDF()


# In[38]:

df.show()


# In[39]:

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

categorical_columns = ['Gender', 'Occupation', 'City_Category', 'Marital_Status']

indexers = [
    StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
    for c in categorical_columns
]

encoders = [OneHotEncoder(dropLast=False, inputCol=indexer.getOutputCol(),
                          outputCol="{0}_encoded".format(indexer.getOutputCol()))
            for indexer in indexers
            ]

assembler = [VectorAssembler(inputCols=[encoder.getOutputCol()
                                        for encoder in encoders], outputCol="features")]

pipeline = Pipeline(stages=indexers + encoders + assembler)

model = pipeline.fit(df)

transformed = model.transform(df)

transformed.select('features').show(5)


# In[40]:

transformed.select('features').take(4)


# ### Example 2
#
# https://stackoverflow.com/questions/32982425/encode-and-assemble-multiple-features-in-pyspark

# In[41]:

from pyspark.sql import Row
from pyspark.mllib.linalg import DenseVector

row = Row("gender", "foo", "bar")

df = sc.parallelize([
    row("0", 3.0, DenseVector([0, 2.1, 1.0])),
    row("1", 1.0, DenseVector([0, 1.1, 1.0])),
    row("1", -1.0, DenseVector([0, 3.4, 0.0])),
    row("2", -3.0, DenseVector([0, 4.1, 0.0]))
]).toDF()


# In[42]:

df.take(4)


# #### StringIndexer

# In[43]:

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="gender", outputCol="gender_numeric").fit(df)
indexed_df = indexer.transform(df)
indexed_df.show()


# #### OneHotEncoder

# In[44]:

from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCol="gender_numeric", outputCol="gender_vector")
encoded_df = encoder.transform(indexed_df)
encoded_df.show()


# In[45]:

encoded_df.select('gender', 'gender_vector').take(10)


# #### VectorAssembler

# In[46]:

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["gender_vector", "bar", "foo"], outputCol="features")

# encoded_df_with_indexed_bar = (vector_indexer.fit(encoded_df).transform(encoded_df))

final_df = assembler.transform(encoded_df)

final_df.show()


# Note: The third row is a 'SparseVector'. It is the same as `[0, 0, 3.4, 0, -1]`. The output `SparseVector(5, {2: 3.4, 4: -1.0})` means:
#
# - There are 5 elements in the vector.
# - The 2nd (start with 0) element is 3.4.
# - The 4th element is -1.0.
# - All other element in the vector is 0.

# In[47]:

final_df.select('features').take(4)


# #### Finally you can wrap all of that using pipelines

# In[48]:

df.show()


# In[49]:

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[indexer, encoder, assembler])
model = pipeline.fit(df)
transformed = model.transform(df)


# In[50]:

transformed.show()


# In[51]:

transformed.select('features').take(4)


# ### Authors
# Quan Cai, data scientist, IBM Watson Platform for Health.
#
# Date: 11/08/2017
