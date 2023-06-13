import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("Ch10 - ex10").getOrCreate()
gsod = spark.read.parquet("/opt/spark/data/window/gsod.parquet")


# ex10.1
every_day = Window.partitionBy("year", "mo", "da")

(
    gsod.withColumn("max_temp", F.max("temp").over(every_day))
    .select("stn", "year", "mo", "da", "temp")
    .where("temp == max_temp")
    .orderBy("year", "mo", "da")
    .show(5)
)


# ex10.2
# It will try to split the records of each window into N equal (as much as possible) buckets
exo10_2 = spark.createDataFrame([[x // 4, 2] for x in range(10)], ["index", "value"])
exo10_2.show()


sol10_2 = Window.partitionBy("index").orderBy("value")

exo10_2.withColumn("10_2", F.ntile(3).over(sol10_2)).show(10)


# ex10.3
exo10_3 = spark.createDataFrame([[10] for x in range(1_000_001)], ["ord"])

exo10_3.select(
    "ord",
    F.count("ord")
    .over(Window.partitionBy().orderBy("ord").rowsBetween(-2, 2))
    .alias("row"),
    F.count("ord")
    .over(Window.partitionBy().orderBy("ord").rangeBetween(-2, 2))
    .alias("range"),
).show(10)

# ex10.4
each_year = Window.partitionBy("year")

(
    gsod.withColumn("max_temp", F.max("temp").over(each_year))
    .where("temp = max_temp")
    .select("year", "mo", "da", "stn", "temp")
    .withColumn("avg_temp", F.avg("temp").over(each_year))
    .orderBy("year", "mo", "da")
    .show()
)


# ex10.5
temp_per_month_asc = Window.partitionBy("mo").orderBy("count_temp")

gsod_light = spark.read.parquet("/opt/spark/data/window/gsod_light.parquet")
gsod_light.withColumn("rank_tpm", F.rank().over(temp_per_month_asc)).show()


temp_per_month_rnk = Window.partitionBy("mo").orderBy("count_temp", "row_tpm")

gsod_light.withColumn("row_tpm", F.row_number().over(temp_per_month_asc)).withColumn(
    "rank_tpm", F.rank().over(temp_per_month_rnk)
).show()


# ex10.6
seven_day_window = (
    Window.partitionBy("stn")
    .orderBy("dtu")
    .rangeBetween(-7 * 24 * 60 * 60, 7 * 24 * 60 * 60)
)

gsod.select(
    "stn", "temp", (F.to_date(F.concat_ws("-", "year", "mo", "da"))).alias("dt")
).withColumn("dtu", F.unix_timestamp("dt").alias("dtu")).withColumn(
    "max_temp", F.max("temp").over(seven_day_window)
).where(
    "temp == max_temp"
).show(
    10
)


# ex10.7
gsod_light_p = gsod_light.withColumn("year", F.lit(2019))
gsod_light_p.show()


one_month = Window.partitionBy("year").orderBy("mo_idx").rangeBetween(-1, 1)

gsod_light_p.withColumn(
    "mo_idx", F.col("year").cast("int") * 12 + F.col("mo").cast("int")
).withColumn("avg_count", F.avg("count_temp").over(one_month)).show()


spark.stop()
