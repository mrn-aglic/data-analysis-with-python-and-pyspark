import pyspark.sql.functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Ch10 - coldest day").getOrCreate()

gsod = spark.read.parquet("/opt/spark/data/window/gsod.parquet")


coldest_temp = gsod.groupby("year").agg(F.min("temp").alias("temp"))
coldest_temp.orderBy("temp").show()

coldest_when = gsod.join(coldest_temp, how="left_semi", on=["year", "temp"]).select(
    "stn", "year", "mo", "da", "temp"
)  # this is basically a self-join - joining the table with something that is already in the table (comes from the table)

coldest_when.orderBy("year", "mo", "da").show()


# using window function
from pyspark.sql.window import Window

each_year = Window.partitionBy("year")  # a blueprint for the window function


# apply the window function
(
    gsod.withColumn("min_temp", F.min("temp").over(each_year))
    .where("temp == min_temp")
    .select("year", "mo", "da", "stn", "temp")
    .orderBy("year", "mo", "da")
    .show()
)


spark.stop()
