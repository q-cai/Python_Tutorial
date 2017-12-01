# coding: utf-8

# Search for the file in file HDFS reservior.

# In[1]:

get_ipython().system(u'hadoop fs -ls /files/lsf/w2-cea6f05ca9994c1dbfe596f1bafcca11')


# Import relevant modules.

# In[35]:

import numpy as np
from PIL import Image
from StringIO import StringIO

get_ipython().magic(u'matplotlib inline')
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Read in binary file with `binaryFiles`.

# In[3]:

images = sc.binaryFiles("/files/lsf/w2-cea6f05ca9994c1dbfe596f1bafcca11")


# The raw data for a PNG image having the form of `\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x0....`

# In[27]:

rawdata = images.take(1)[0][1]


# Create a PIL image from the raw data, and read it as a nparray.

# In[29]:

a = np.asarray(Image.open(StringIO(rawdata)))


# The shape of the image: pixel length, pixle width and color (CMYK?).

# In[30]:

a.shape


# The image is now in a numpy array format.

# In[33]:

type(a)


# Plot the image file.

# In[36]:

plt.imshow(a)
plt.show()


# Print the whole raw data of the png image.

# In[37]:

# images.take(1)


# ### Reference:
#
# [1] https://stackoverflow.com/questions/33138129/spark-using-pyspark-read-images

# ### Authors
#
# Quan Cai, data scientist, IBM Watson Platform for Health.
# Date: 12/01/2017
