#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys 
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from timeit import default_timer as timer
     
def already_trained(rank, reg, alpha):
    return True

# Only enter this block if we're in main
def main(spark, user):
    # fname = 'als_rank_%d_reg_%s_alpha_%s' % (10, .1, 1.0)
    # model = ALSModel.load('hdfs:/user/%s/%s' % (user, fname))
    model = ALSModel.load('hdfs:/user/%s/als_test' % user)
    val_df = spark.read.parquet('hdfs:/user/%s/val_full_indexed.parquet' % user)
    predictions = model.transform(val_df)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

# Only enter this block if we're in main
if __name__ == "__main__":
    start = timer()
    # Create the spark session object
    spark = SparkSession.builder.appName('train').getOrCreate()

    user = sys.argv[1]

    # Call our main routine
    main(spark, user)
    end = timer()
    print(end - start)