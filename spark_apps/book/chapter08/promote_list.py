from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Ch08 - promote list").getOrCreate()

collection = [1, "two", 3.0, ("four", 4), {"five", 5}]

sc = spark.sparkContext

collection_rdd = sc.parallelize(collection)  # promote list to RDD

print(collection_rdd)
