# %%
import numpy as np
import pandas as pd

import boto3

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, PCA

import cv2 as cv

local = True
write_data = True
# %%
spark = SparkSession.builder.appName('FruitsPreProc').config(
    'spark.hadoop.fs.s3a.impl',
    'org.apache.hadoop.fs.s3a.S3AFileSystem').getOrCreate()
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
s3 = boto3.client('s3')

# %%
if local is True:
    path = './fruits-360_dataset/fruits-360/Training/'
else:
    path = 's3a://stockp8oc/fruits-360/Training/'


def load_img_data(path=path):
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
    pcaDataDFClean.to_csv('./featuresPCA.csv', index=False)
# %%
