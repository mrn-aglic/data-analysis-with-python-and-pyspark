from pyspark.sql import SparkSession
from functools import reduce
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd

spark = SparkSession.builder.appName("CH 09 - series to series UDF").getOrCreate()


gsod = (
    reduce(
        lambda x, y: x.unionByName(y, allowMissingColumns=True),
        [
            spark.read.parquet(f"/opt/spark/data/gsod_noaa/gsod{year}.parquet")
            for year in range(2019, 2021)
        ],
    )
    .dropna(subset=["year", "mo", "da", "temp"])
    .where(F.col("temp") != 9999.9)
    .drop("date")
)

gsod.show(5)

@F.pandas_udf(T.DoubleType())
def f_to_c(degrees: pd.Series) -> pd.Series:
    return (degrees - 32) * 5 / 9


gsod = gsod.withColumn("temp_c", f_to_c(F.col("temp")))
gsod.select("temp", "temp_c").distinct().show(5)


spark.stop()
