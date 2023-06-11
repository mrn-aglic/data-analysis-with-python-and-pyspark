from functools import reduce

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Ch09 - ex09").getOrCreate()

# ex 9.1
ex9_1 = pd.Series(["red", "blue", "blue", "yellow"])


# pylint: disable=unnecessary-lambda
def color_to_num(colors: pd.Series) -> pd.Series:
    return colors.apply(lambda x: {"red": 1, "blue": 2, "yellow": 3}.get(x))


color_to_num(ex9_1)


color_to_num_udf = F.pandas_udf(color_to_num, T.IntegerType())
ex9_1_df = spark.createDataFrame(ex9_1.to_frame())

ex9_1_df.show()


ex9_1_df.select(color_to_num_udf(F.col("0")).alias("num")).show(5)


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


# ex 9.2
def temp_to_temp(value: pd.Series, from_temp: str, to_temp: str) -> pd.Series:
    from_temp = str.upper(from_temp)
    to_temp = str.upper(to_temp)

    acceptable_values = ["F", "C", "R", "K"]
    if to_temp not in acceptable_values or from_temp not in acceptable_values:
        return value.apply(lambda _: None)

    from_to = {
        ("C", "F"): lambda: value * (9 / 5) + 32,
        ("F", "C"): lambda: (value - 32) * (5 / 9),
        ("C", "K"): lambda: value + 273.15,
        ("K", "C"): lambda: value - 273.15,
        ("C", "R"): lambda: value * (9 / 5) + 491.67,
        ("R", "C"): lambda: (value - 491.67) * (5 / 9),
        ("F", "K"): lambda: (value - 32) * (5 / 9) + 273.15,
        ("K", "F"): lambda: (value - 273.15) * (9 / 5) + 32,
        ("F", "R"): lambda: value + 459.67,
        ("R", "F"): lambda: value - 459.67,
        ("K", "R"): lambda: value * (9 / 5),
        ("R", "K"): lambda: value * (5 / 9),
    }

    convert = from_to[(from_temp, to_temp)]
    return convert()


gsod.select("temp", temp_to_temp(F.col("temp"), "F", "C").alias("temp_c")).show(
    5, False
)


# ex 9.3
def scale_temperature_c(temp_by_day: pd.DataFrame) -> pd.DataFrame:
    def f_to_c_temp(temp):
        return (temp - 32.0) * 5.0 / 9.0

    temp = f_to_c_temp(temp_by_day.temp)
    answer = temp_by_day[["stn", "year", "mo", "da", "temp"]]
    if temp.min() == temp.max():
        return answer.assign(temp_norm=0.5)
    return answer.assign(temp_norm=(temp - temp.min()) / (temp.max() - temp.min()))


# ex 9.4
gsod_ex = (
    gsod.groupby("year", "mo")
    .applyInPandas(
        scale_temperature_c,
        schema=T.StructType(
            [
                T.StructField("stn", T.StringType()),
                T.StructField("year", T.StringType()),
                T.StructField("mo", T.StringType()),
                T.StructField("da", T.StringType()),
                T.StructField("temp", T.DoubleType()),
                T.StructField("temp_norm", T.DoubleType()),
            ]
        ),
    )
    .show(5, False)
)


from typing import Sequence

# ex 9.5
from sklearn.linear_model import LinearRegression


@F.pandas_udf(T.ArrayType(T.DoubleType()))
def rate_of_change_temp(day: pd.Series, temp: pd.Series) -> Sequence[float]:
    fitted = LinearRegression().fit(X=day.astype("int").values.reshape(-1, 1), y=temp)

    return fitted.coef_[0], fitted.intercept_


result = gsod.groupby("stn", "year", "mo").agg(
    rate_of_change_temp(gsod["da"], gsod["temp"]).alias("sol_9_5")
)

result.show(5, False)


spark.stop()
