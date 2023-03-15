from fractions import Fraction
from typing import Optional, Tuple

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Ch08 - UDF").getOrCreate()


fractions = [[x, y] for x in range(100) for y in range(1, 100)]
frac_df = spark.createDataFrame(fractions, ["numenator", "denominator"])


frac_df = frac_df.select(
    F.array(F.col("numenator"), F.col("denominator")).alias("fraction")
)

frac_df.show(5, False)

Frac = Tuple[int, int]  # type synonym


def py_reduce_fraction(frac: Frac) -> Optional[Frac]:
    """Reduce a fraction represented as a 2-tuple of integers."""
    num, denom = frac

    if denom:
        answer = Fraction(num, denom)
        return answer.numerator, answer.denominator
    return None


assert py_reduce_fraction((3, 6)) == (1, 2)
assert py_reduce_fraction((1, 0)) is None


def py_fraction_to_float(frac: Frac) -> Optional[float]:
    """Transform a fraction represented as a 2-tuple of integers into a float."""
    num, denom = frac

    if denom:
        return num / denom
    return None


assert py_fraction_to_float((2, 8)) == 0.25
assert py_fraction_to_float((1, 0)) is None


SparkFrac = T.ArrayType(T.LongType())

reduce_fraction = F.udf(py_reduce_fraction, SparkFrac)

frac_df = frac_df.withColumn("reduced_fraction", reduce_fraction(F.col("fraction")))

frac_df.show(5, False)


@F.udf(T.DoubleType())
def fraction_to_float(frac: Frac) -> Optional[float]:
    """Transform a fraction represented as a 2-tuple of integers as a float."""
    num, denom = frac
    if denom:
        return num / denom
    return None


frac_df = frac_df.withColumn(
    "fraction_float", fraction_to_float(F.col("reduced_fraction"))
)

frac_df.select("reduced_fraction", "fraction_float").distinct().show(5, False)
