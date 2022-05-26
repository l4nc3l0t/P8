# %%
import numpy as np
import pandas as pd

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType

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
    ImgData.show(3, truncate=False)
    return ImgData.sample(.1)


# %%
ImgData = load_img_data()
"""
# %%
iv3 = inception_v3.InceptionV3(weights='imagenet', include_top=False)
img = image.load_img(
    './fruits-360_dataset/fruits-360/Training/Pineapple Mini/170_100.jpg',
    target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = inception_v3.preprocess_input(x)
desc = iv3.predict(x).flatten()
print(desc)


# %%
def get_desc(path):
    iv3 = inception_v3.InceptionV3(weights='imagenet', include_top=False)
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception_v3.preprocess_input(x)
    desc = iv3.predict(x)
    print(desc)

    return desc


# %%
udf_image = F.udf(get_desc, ArrayType(FloatType(), containsNull=False))

ImgDesc = ImgData.withColumn("descriptors", udf_image("TruePath"))

ImgDesc = ImgDesc.filter(ImgDesc.descriptors.isNotNull())

ImgDesc.select('descriptors').collect()  #.show(3, truncate=False)

"""
# %%
def get_desc(img):
    image = cv.imread(img)
    orb = cv.ORB_create(nfeatures=100)
    keypoints_orb, desc = orb.detectAndCompute(image, None)
    #if desc is None:
    #    desc = np.array(32 * [0])
    #else:
    #    desc = desc
    return desc


# %%
def make_bovw():
    Desc = {}
    BoVW = []
    for p, n in zip(
            ImgData.select('TruePath').toPandas().TruePath,
            ImgData.select('imgName').toPandas().imgName):
        desc = get_desc(p)

        Desc[n] = desc
        if len(BoVW) == 0:
            BoVW = desc
        elif desc is None:
            BoVW = np.vstack((BoVW, 32 * [0]))
        else:
            BoVW = np.vstack((BoVW, desc))
    BoVW = np.float32(BoVW)

    idx = []
    for i in ImgData.select('imgName').toPandas().imgName:
        if Desc[i] is None:
            idx.extend([i])
        else:
            idx.extend(len(Desc[i]) * [i])
    # Clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
    flags = cv.KMEANS_PP_CENTERS
    compactness, labels, centers = cv.kmeans(BoVW, 1000, None, criteria, 1,
                                             flags)

    Lab = pd.DataFrame(
        labels.ravel(), index=idx,
        columns=['label']).reset_index().rename(columns={'index': 'ImgName'})
    Lab['Fruit'] = Lab.ImgName.str.split('_', 1, expand=True).drop(columns=[1])

    LabClean = Lab.drop(
        columns='ImgName').groupby('Fruit').value_counts().reset_index().pivot(
            index='Fruit', columns='label', values=0).fillna(0)
    LabClean.head(5)
    return LabClean


# %%
BoVW = make_bovw()

# %%
