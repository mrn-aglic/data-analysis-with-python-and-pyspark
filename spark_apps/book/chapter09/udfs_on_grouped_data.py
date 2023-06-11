from functools import reduce

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from sklearn.linear_model import LinearRegression

spark = SparkSession.builder.appName("CH 09 - UDF on grouped data").getOrCreate()

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


# also called Series to Scalar UDF
@F.pandas_udf(T.DoubleType())
def rate_of_change_temp(day: pd.Series, temp: pd.Series) -> float:
    return (
        LinearRegression().fit(X=day.astype(int).values.reshape(-1, 1), y=temp).coef_[0]
    )


result = gsod.groupby("stn", "year", "mo").agg(
    rate_of_change_temp(gsod["da"], gsod["temp"]).alias("rt_change_temp")
)

result.show(5, False)


# the group map pattern: maps over each batch and returns a DataFrame which are later combined into a single DataFrame
# the function must return a complete DataFrame, which means that all of the columns need to be returned -
# including the ones we groupped by
# the pandas_udf decorator is NOT NEEDED in this case
def scale_temperature(temp_by_day: pd.DataFrame) -> pd.DataFrame:
    temp = temp_by_day.temp
    answer = temp_by_day[["stn", "year", "mo", "da", "temp"]]

    if temp.min() == temp.max():
        return answer.assign(temp_norm=0.5)

    return answer.assign(temp_norm=(temp - temp.min()) / (temp.max() - temp.min()))


schema = T.StructType(
    [
        T.StructField("stn", T.StringType()),
        T.StructField("year", T.StringType()),
        T.StructField("mo", T.StringType()),
        T.StructField("da", T.StringType()),
        T.StructField("temp", T.DoubleType()),
        T.StructField("temp_norm", T.DoubleType()),
    ]
)


gsod_map = gsod.groupby("stn", "year", "mo").applyInPandas(
    scale_temperature, schema=schema
)


gsod_map.show(5, False)
