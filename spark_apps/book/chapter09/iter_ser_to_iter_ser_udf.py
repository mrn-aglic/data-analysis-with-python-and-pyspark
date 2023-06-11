from functools import reduce
from typing import Iterator

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName(
    "CH 09 - iter series to iter series UDF"
).getOrCreate()

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


from time import sleep


# explicitly iterating over each batch one by one
@F.pandas_udf(T.DoubleType())
def f_to_c(degrees: Iterator[pd.Series]) -> Iterator[pd.Series]:
    sleep(5)  # simulate cold start that happens on each worker once
    for batch in degrees:
        yield (batch - 32) * 5 / 9


gsod.select("temp", f_to_c(F.col("temp")).alias("temp_c")).distinct().show(5)


from typing import Tuple


# the iterator of multiple series to iterator of series
@F.pandas_udf(T.DateType())
def create_date(
    year_mo_da: Iterator[Tuple[pd.Series, pd.Series, pd.Series]]
) -> Iterator[pd.Series]:
    for year, mo, da in year_mo_da:
        yield pd.to_datetime(pd.DataFrame(dict(year=year, month=mo, day=da)))


# pylint: disable=too-many-function-args
gsod.select(
    "year",
    "mo",
    "da",
    create_date(F.col("year"), F.col("mo"), F.col("da")).alias("date"),
).distinct().show(5)

spark.stop()
