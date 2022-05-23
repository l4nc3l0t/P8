# %%
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, functions as F

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
    ImgData = ImgData.drop('modificationTime', 'length')
    ImgData.show(3)

# %%
