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
spark = SparkSession.builder.appName('FruitsPreProc').config(
    "spark.driver.memory", "8g").getOrCreate()
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
    return ImgData.sample(.1)


# %%
ImgData = load_img_data()


# %%
def get_desc(img):
    image = cv.imread(img)
    if image is not None:
        imgR = cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)
        imgBW = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        imgBWCLAHE = cv.createCLAHE(clipLimit=8,
                                    tileGridSize=(3, 3)).apply(imgBW)
        imgBWCLAHENlMD = cv.fastNlMeansDenoising(imgBWCLAHE, None, 5, 7, 21)
        orb = cv.ORB_create(nfeatures=100)
        keypoints_orb, desc = orb.detectAndCompute(imgBWCLAHENlMD, None)
        if desc is None:
            desc = [np.array(32 * [0]).astype(np.float64).tolist()]
        else:
            desc = desc.astype(np.float64).tolist()
    else:
        desc = [np.array(32 * [0]).astype(np.float64).tolist()]

    return desc


# %%
udf_image = F.udf(
    get_desc,
    ArrayType(ArrayType(FloatType(), containsNull=False), containsNull=False))

ImgDesc = ImgData.drop('content').withColumn("descriptors",
                                             F.explode(udf_image("TruePath")))

ImgDesc = ImgDesc.filter(ImgDesc.descriptors.isNotNull())

# %%
model = KMeans(k=1000,
               featuresCol='descriptors').fit(ImgDesc)
# %%
ImgPred = model.transform(ImgDesc)
ImgPred.show(3)
# %%
ImgPred = ImgPred.groupBy('label', 'prediction').count().fillna(0)
#%%
BoVW = ImgPred.groupBy('label').pivot('prediction').show(3)

# %%
