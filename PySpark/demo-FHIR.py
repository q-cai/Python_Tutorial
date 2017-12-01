
# coding: utf-8

# ## Table of Contents
#
# 1. [Introduction](#context1)<br>
# 2. [Spark Setup](#context2)<br>
#     2.1 [Import SQLContext to initialize the Spark context](#context21)<br>
#     2.2 [Search for files in FHIR HDFS with root "/data/lsf/lsf_appname="](#context22)<br>
# 3. [Retrieve information from MedicationAdministration](#context3)<br>
# 4. [Retrieve information from QuestionnaireResponse](#context4)<br>
# 5. [Retrieve information from Observation](#context5)<br>

# <a id="context1"></a>
# ## 1. Introduction
#
# In this notebook, we show a demo to retrieve information form data reservior - FHIR (HDFS). Specifically, we retrieve information about __MedicationAdministration__, __QuestionnaireResponse__ and __Observation__.

# <a id="context2"></a>
# ## 2. Spark Setup
#
# <a id="context21"></a>
# ### 2.1. Import SQLContext to initialize the Spark context

# In[1]:

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


# <a id="context22"></a>
# ### 2.2. Search for files in FHIR HDFS with root "`/data/lsf/lsf_appname=`"

# In[2]:

get_ipython().system(u'hadoop fs -ls /data/lsf/lsf_appname=*')


# <a id="context3"></a>
# ## 3. Retrieve information from MedicationAdministration

# In[3]:

med_admin_path = "/data/lsf/lsf_appname=null/lsf_resourcetype=MedicationAdministration/"
med_admin_df = sqlContext.read.parquet(med_admin_path)
med_admin_df.schema.names


# In[4]:

# select the columns (features) of our interest only
med_admin_df = med_admin_df.select(
    "appName", "deviceInfo", "resourceType", "status", "lsf_year", "lsf_month", "lsf_day")

print("There are %d records retrieved from %s" % (med_admin_df.count(), med_admin_path))

med_admin_df.show(5, False)


# <a id="context4"></a>
# ## 4. Retrieve information from QuestionnaireResponse

# In[5]:

que_res_path = "/data/lsf/lsf_appname=null/lsf_resourcetype=QuestionnaireResponse"
que_res_df = sqlContext.read.parquet(que_res_path)
que_res_df.schema.names


# In[6]:

# select the columns (features) of our interest only
que_res_df = que_res_df.select("appName", "deviceInfo", "resourceType",
                               "status", "lsf_year", "lsf_month", "lsf_day")

print("There are %d records retrieved from %s" % (que_res_df.count(), que_res_path))

que_res_df.show(5, False)


# <a id="context5"></a>
# ## 5. Retrieve information from Observation

# In[7]:

obs_path = "/data/lsf/lsf_appname=null/lsf_resourcetype=Observation"
obs_df = sqlContext.read.parquet(obs_path)
obs_df.schema.names


# In[8]:

# select the columns (features) of our interest only
obs_df = obs_df.select("appName", "deviceInfo", "resourceType",
                       "status", "lsf_year", "lsf_month", "lsf_day")

print("There are %d records retrieved from %s" % (obs_df.count(), obs_path))

obs_df.show(5, False)


# ### Authors
# Quan Cai, data scientist, IBM Watson Platform for Health.

# In[ ]:
