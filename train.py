#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys 
import random 
import itertools
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
     
def already_trained(rank, reg, alpha):
    return True

# Only enter this block if we're in main
def main(spark, user):
    train_df = spark.read.parquet('hdfs:/user/jm7955/sub_train.parquet')
    path = 'hdfs:/user/%s/' % user

    ranks = (8, 10, 12)
    regs = (.01, .1, 1)
    alphas = (1.0, 2.0, 3.0, 4.0)
    params_list = list(itertools.product(ranks, regs, alphas))
    random.shuffle(params_list)

    rank, reg, alpha = (10, .1, 1.0)
    als = ALS(regParam=reg, rank=rank, alpha=alpha, ratingCol="count")
    model = als.fit(train_df)
    #model.save(path + 'als_test')
    model.save(path + 'als_rank_%d_reg_%s_alpha_%s' % (10, .1, 1.0))

    for params in params_list:
        rank, reg, alpha = params
        if already_trained(rank, reg, alpha):
            continue
        als = ALS(regParam=reg, rank=rank, alpha=alpha, ratingCol="count")
        model = als.fit(train_df)
        model.save(path + 'als_rank_%d_reg_%s_alpha_%s' % params)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('train').getOrCreate()

    user = sys.argv[1]



    # Call our main routine
    main(spark, user)