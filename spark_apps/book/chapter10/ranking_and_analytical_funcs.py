import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("Ch10 - ranking and analytical").getOrCreate()
gsod_light = spark.read.parquet("/opt/spark/data/window/gsod_light.parquet")


temp_per_month_asc = Window.partitionBy("mo").orderBy("count_temp")

# rank
gsod_light.withColumn("rank_tpm", F.rank().over(temp_per_month_asc)).show()

# dense_rank
gsod_light.withColumn("rank_tpm", F.dense_rank().over(temp_per_month_asc)).show()


# percent_rank
# For every window, percent_rank() will compute the percentage rank (between zero and one) based on the ordered value.
temp_each_year = Window.partitionBy("year").orderBy("temp")

gsod_light.withColumn("rank_tpm", F.percent_rank().over(temp_each_year)).show()

# percent_rank_formula: number_of_rows_with_value_lower_than_current_one / (number_of_rows_in_the_window - 1)


# ntile
# Each value is placed into a tile, which one depends on the value and number of tiles
gsod_light.withColumn("rank_tpm", F.ntile(2).over(temp_each_year)).show()

# row_number
gsod_light.withColumn("row_num", F.row_number().over(temp_each_year)).show()

# descending
temp_per_month_desc = Window.partitionBy("mo").orderBy(F.col("count_temp").desc())

gsod_light.withColumn("row_num", F.row_number().over(temp_per_month_desc)).show()


# # Analytic functions
gsod_light.withColumn("previous_temp", F.lag("temp").over(temp_each_year)).withColumn(
    "previous_temp2", F.lag("temp", 2).over(temp_each_year)
).show()


# cume_dist formula: num_rows_with_lower_equal_value_than_current_one / num_rows_in_window
gsod_light.withColumn("percent_rank", F.percent_rank().over(temp_each_year)).withColumn(
    "cume_dist", F.cume_dist().over(temp_each_year)
).show()


spark.stop()
