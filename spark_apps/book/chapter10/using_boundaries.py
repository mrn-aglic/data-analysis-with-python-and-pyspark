import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("Ch10 - using boundaries").getOrCreate()
gsod_light = spark.read.parquet("/opt/spark/data/window/gsod_light.parquet")


not_ordered = Window.partitionBy("year")
ordered = not_ordered.orderBy("temp")

gsod_light.withColumn("avg_NO", F.avg("temp").over(not_ordered)).withColumn(
    "avg_O", F.avg("temp").over(ordered)
).show()


# With no ordering, the Spark API specifies an unbounded window frame - rowFrame, unbounded preceding, unbounded following.
# With ordering, the Spark API uses a growing window frame - rangeFrame, unbounded preceding, current row

not_ordered = Window.partitionBy("year").rowsBetween(
    Window.unboundedPreceding, Window.unboundedFollowing
)
ordered = not_ordered.orderBy("temp").rangeBetween(
    Window.unboundedPreceding, Window.currentRow
)


# If your window spec is not ordered, using a boundary is a nondeterministic operation. Spark will not guarantee that your window will contain the same values as we are not ordering within a window before picking the boundary. This also applies if you order the data frame in a previous operation. If you use a boundary, provide an explicit ordering clause.


gsod_light_p = (
    gsod_light.withColumn("year", F.lit(2019))
    .withColumn(
        "dt",
        F.to_date(F.concat_ws("-", F.col("year"), F.col("mo"), F.col("da"))),
    )
    .withColumn("dt_num", F.unix_timestamp("dt"))
)
gsod_light_p.show()


ONE_MONTH_ISH = 30 * 60 * 60 * 24  # or 2_592_000 seconds
one_month_ish_before_and_after = (
    Window.partitionBy("year")
    .orderBy("dt_num")
    .rangeBetween(-ONE_MONTH_ISH, ONE_MONTH_ISH)
)
gsod_light_p.withColumn(
    "avg_count", F.avg("count_temp").over(one_month_ish_before_and_after)
).show()


spark.stop()
