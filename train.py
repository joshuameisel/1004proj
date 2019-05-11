#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys 
import random 
import itertools
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from timeit import default_timer as timer
     
def already_trained(rank, reg, alpha):
    return True

# Only enter this block if we're in main
def main(spark, user, alpha):
    train_df = spark.read.parquet('hdfs:/user/jm7955/train_full_indexed.parquet')
    path = 'hdfs:/user/%s/' % user

    ranks = (10, 20, 30, 40)
    regs = (.001, .01, .1, 1)
    params_list = list(itertools.product(ranks, regs))
    random.shuffle(params_list)

    for params in params_list:
        rank, reg = params
        if already_trained(rank, reg, alpha):
            continue

        print('starting with alpha=%s, rank=%d, reg=%s' % (rank, reg, alpha))
        start = timer()
        als = ALS(regParam=reg, rank=rank, alpha=alpha, ratingCol="count")
        model = als.fit(train_df)
        model.save(path + 'als_rank_%d_reg_%s_alpha_%s' % params)
        print('finished in %s seconds' % timer() - start)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('train').getOrCreate()

    user = sys.argv[1]

    alpha = sys.argv[2]

    # Call our main routine
    main(spark, user, float(alpha))