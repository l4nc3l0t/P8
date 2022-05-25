# %%
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.image import ImageSchema
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.ml.clustering import KMeans

import cv2 as cv

# %%
spark = SparkSession.builder.appName('FruitsPreProc').getOrCreate()
sc = spark.sparkContext


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
    #img2vec = F.udf(lambda x: DenseVector(ImageSchema.toNDArray(x).flatten()),
    #                VectorUDT())
    #ImgData = ImgData.withColumn('vecs', img2vec("TruePath"))
    ImgData.show(3)
    return ImgData

# %%
ImgData = load_img_data()

# %%
def get_desc(img):

    image = cv.imread(img)
    sift = cv.SIFT_create(nfeatures=100)
    keypoints_sift, desc = sift.detectAndCompute(image, None)

    if desc is None:

        desc = 0
    else:
        desc = desc.flatten().tolist()

    return desc

udf_image = F.udf(get_desc, ArrayType(FloatType(), containsNull=False))

ImgDesc = ImgData.withColumn("descriptors", udf_image("TruePath"))

ImgDesc = ImgDesc.filter(ImgDesc.descriptors.isNotNull())

ImgDesc.show(3)
# %%
model = KMeans(k=100, featuresCol='descriptors').fit(ImgDesc)
# %%
"""
def make_bovw(Desc):

BoVW = make_bovw(ImgDesc.select('descriptors'))
"""
# %%
