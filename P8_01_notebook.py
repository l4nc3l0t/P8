# %%
import numpy as np
import pandas as pd

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, PCA

import io
from PIL import Image
import cv2 as cv

local = True
write_data = True

import os

os.environ[
    'PYSPARK_SUBMIT_ARGS'] = '--packages com.amazonaws:aws-java-sdk:1.12.230,org.apache.hadoop:hadoop-aws:3.3.1 pyspark-shell'

# %%
spark = SparkSession.builder.master('local').appName(
    'FruitsPreProc').getOrCreate()
#.config(
#"spark.hadoop.fs.s3a.aws.credentials.provider",
#"com.amazonaws.auth.profile.ProfileCredentialsProvider").config(
#'spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')

sc = spark.sparkContext

sc._jsc.hadoopConfiguration().set('fs.s3a.impl',
                                  'org.apache.hadoop.fs.s3a.S3AFileSystem')
sc._jsc.hadoopConfiguration().set(
    "fs.s3a.aws.credentials.provider",
    "com.amazonaws.auth.profile.ProfileCredentialsProvider")
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint",
                                  "s3.eu-west-3.amazonaws.com")

spark.sparkContext._conf.getAll()

# %%
if local is True:
    path = './fruits-360_dataset/fruits-360/Training/'
else:
    #path = 's3a://stockp8oc/fruits-360/LightTrain/'
    path = 's3a://stockp8oc/fruits-360/Training/'


ImgData = spark.read.format('binaryFile') \
                .option('pathGlobFilter', '*.jpg') \
                .option('recursiveFileLookup', 'true') \
                .load(path) \
                .select('path', 'content')
ImgData = ImgData.withColumn('label',
                             F.element_at(F.split(F.col('path'), '/'), -2))
if local is True:
    ImgData = ImgData.withColumn('TruePath',
                                 F.element_at(F.split(F.col('path'), ':'), 2))
else:
    ImgData = ImgData.withColumn('TruePath', F.col('path'))

ImgData = ImgData.withColumn(
    'imgName',
    F.concat('label', F.lit('_'), F.element_at(F.split(F.col('path'), '/'),
                                               -1)))
ImgData = ImgData.drop('path')


# %%
def get_desc(content):
    try:
        img = np.array(Image.open(io.BytesIO(content)))
    except:
        print(content)
        img = None
        return img
    if img is None:
        desc = None
    else:
        orb = cv.ORB_create(nfeatures=100)
        keypoints_orb, desc = orb.detectAndCompute(img, None)
    if desc is None:
        desc = [np.array(32 * [0]).astype(np.float64).tolist()]
    else:
        desc = desc.astype(np.float64).tolist()
    return desc


# %%
udf_image = F.udf(
    get_desc,
    ArrayType(ArrayType(FloatType(), containsNull=False), containsNull=False))

ImgDesc = ImgData.withColumn("descriptors", F.explode(udf_image("content")))
ImgDesc.show(3)
# %%
kmean = KMeans(k=1000, featuresCol='descriptors', seed=0)
model = kmean.fit(ImgDesc)

# %%
Pred = model.transform(ImgDesc)
Pred.show(3)
# %%
ImgPred = Pred.groupBy('label', 'prediction').count()
#%%
BoVW = ImgPred.groupBy('label').pivot('prediction').sum('count').fillna(0)
BoVW.show()

# %%
VA = VectorAssembler(inputCols=BoVW.drop('label').columns,
                     outputCol='features')
pca = PCA(k=100, inputCol='features', outputCol='pca_features')
pipe = Pipeline(stages=[VA, pca])
# %%
pipePCA = pipe.fit(BoVW)
# %%
pcaData = pipePCA.transform(BoVW)
pcaDataDF = pcaData.select(['label', 'pca_features']).toPandas()
# %%
pcaDataDFClean = pcaDataDF.join(
    pd.DataFrame(
        pcaDataDF['pca_features'].tolist())).drop(columns='pca_features')
if write_data is True:
    if local is True:
        pcaDataDFClean.to_csv('./featuresPCA.csv', index=False)
    else:
        pcaDataDFClean.to_csv('s3://stockp8oc/featuresPCA.csv', index=False)
# %%
