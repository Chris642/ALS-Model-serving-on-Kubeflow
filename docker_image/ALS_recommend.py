import pyspark
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import Row
from pyspark.sql import SparkSession
import numpy as np


class ALS_recommend(object):
    def __init__(self):
        self.sc = pyspark.SparkContext()
        self.spark = SparkSession\
            .builder\
            .appName("ALSMoviePrediction") \
            .getOrCreate()
        self.model = ALSModel.load('ALSModel')

    def predict(self, X, feature_names):
        R = Row("userId")
        my_data = [X[0].astype(int).item()]
        users = self.spark.createDataFrame([R(i) for i in my_data])
        return np.asarray([np.array(["movieid","score"]), np.asarray([x[0] for x in self.model \
                    .recommendForUserSubset(users, 1).select("recommendations").collect()]).flatten()])






