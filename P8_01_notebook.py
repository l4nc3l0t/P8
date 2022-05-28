# %%
import numpy as np
import pandas as pd

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from pyspark.ml.clustering import KMeans

#from tensorflow.keras.applications import inception_v3
#from tensorflow.keras.preprocessing import image

import cv2 as cv

# %%
spark = SparkSession.builder.appName('FruitsPreProc').getOrCreate()
sc = spark.sparkContext
spark.sparkContext._conf.getAll()


# %%
def load_img_data(path='./fruits-360_dataset/fruits-360/Training/'):
    ImgData = spark.read.format('binaryFile') \
                    .option('pathGlobFilter', '*.jpg') \
                    .option('recursiveFileLookup', 'true') \
                    .load(path) \
                    .select('path', 'content')
    ImgData = ImgData.withColumn('label',
                                 F.element_at(F.split(F.col('path'), '/'), -2))
    ImgData = ImgData.withColumn('TruePath',
                                 F.element_at(F.split(F.col('path'), ':'), 2))
    ImgData = ImgData.withColumn(
        'imgName',
        F.concat('label', F.lit('_'),
                 F.element_at(F.split(F.col('path'), '/'), -1)))
    ImgData = ImgData.drop('path')
    return ImgData.sample(.01)


# %%
ImgData = load_img_data()


# %%
def get_desc(img):
    image = cv.imread(img)
    orb = cv.ORB_create(nfeatures=100)
    keypoints_orb, desc = orb.detectAndCompute(image, None)
    if desc is None:
        desc = [np.array(32 * [0]).astype(np.float64).tolist()]
    else:
        desc = desc.astype(np.float64).tolist()
    return desc


# %%
udf_image = F.udf(
    get_desc,
    ArrayType(ArrayType(FloatType(), containsNull=False), containsNull=False))

ImgDesc = ImgData.drop('content').withColumn("descriptors",
                                             F.explode(udf_image("TruePath")))

# %%
kmean = KMeans(k=1000, featuresCol='descriptors', seed=0)
model = kmean.fit(ImgDesc)

# %%
Pred = model.transform(ImgDesc)
Pred.show(3)
# %%
ImgPred = Pred.groupBy('label', 'prediction').count()
#%%
BoVW = ImgPred.groupBy('label').pivot('prediction').sum('count').fillna(
    0)
BoVW.show()

# %%
