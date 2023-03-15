from operator import add

from py4j.protocol import Py4JJavaError
from pyspark.sql import SparkSession


def add_one(value):
    return value + 1


def safer_add_one(value):
    try:
        return value + 1
    except TypeError:
        return value


spark = SparkSession.builder.appName("Ch08 - basic functions RDD").getOrCreate()

collection = [1, "two", 3.0, ("four", 4), {"five", 5}]

sc = spark.sparkContext

collection_rdd = sc.parallelize(collection)  # promote list to RDD

print(collection_rdd)


collection_rdd_mapped = collection_rdd.map(safer_add_one)

try:
    print(collection_rdd_mapped.collect())
except Py4JJavaError:
    pass

collection_rdd_filtered = collection_rdd_mapped.filter(
    lambda elem: isinstance(elem, (float, int))
)

print(collection_rdd_filtered.collect())

collection_rdd = sc.parallelize([4, 7, 9, 1, 3])

print(collection_rdd.reduce(add))
