import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("Ch10 - udf functions with window").getOrCreate()
gsod_light = spark.read.parquet("/opt/spark/data/window/gsod_light.parquet")


gsod_light.show()


# PySpark applies UDF to each window

import pandas as pd


@F.pandas_udf("double")
def median(vals: pd.Series) -> float:
    return vals.median(
        skipna=False
    )  # without skipna=False, the function throws an exception


each_year_NO = Window.partitionBy("year")
each_year_O = each_year_NO.orderBy("mo", "da")

gsod_light.withColumn("median_temp", median("temp").over(each_year_NO)).withColumn(
    "median_temp_g", median("temp").over(each_year_O)
).show()


# Example from pyspark docs
@F.pandas_udf("double")
def mean_udf(v: pd.Series) -> float:
    return v.mean()


df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)], ("id", "v")
)

w = Window.partitionBy("id").orderBy("id", "v")
df.withColumn("mean_v", mean_udf("v").over(w)).show()


spark.stop()


# In[ ]:
