# %%
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.ml.image import ImageSchema
from pyspark.ml.linalg import DenseVector, VectorUDT

import cv2

# %%
spark = SparkSession.builder.appName('FruitsPreProc').getOrCreate()
sc = spark.sparkContext


# %%
def load_img_data(path='./fruits-360_dataset/fruits-360/Training/'):
    ImgData = spark.read.format('binaryFile') \
                    .option('pathGlobFilter', '*.jpg') \
                    .option('recursiveFileLookup', 'true') \
                    .load(path)
    ImgData = ImgData.withColumn('label',
                                 F.element_at(F.split(F.col('path'), '/'), -2))
    ImgData = ImgData.withColumn('TruePath',
                                 F.element_at(F.split(F.col('path'), ':'), 2))
    ImgData = ImgData.drop('modificationTime', 'length')
    #img2vec = F.udf(lambda x: DenseVector(ImageSchema.toNDArray(x).flatten()),
    #                VectorUDT())
    #ImgData = ImgData.withColumn('vecs', img2vec("TruePath"))
    ImgData.show(3)
    return ImgData

# %%
ImgData = load_img_data()

# %%
def get_desc(img):

    image = cv2.imread(img)
    orb = cv2.ORB_create(nfeatures=50)
    keypoints_orb, desc = orb.detectAndCompute(image, None)

    if desc is None:

        desc = 0
    else:
        desc = desc.flatten().tolist()

    return desc

udf_image = F.udf(get_desc, ArrayType(IntegerType()))

ImgDesc = ImgData.withColumn("descriptors", udf_image("TruePath"))

ImgDesc = ImgDesc.filter(ImgDesc.descriptors.isNotNull())

ImgDesc.show()

# %%
