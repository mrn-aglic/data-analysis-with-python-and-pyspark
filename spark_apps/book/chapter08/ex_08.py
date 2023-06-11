from fractions import Fraction
from operator import add
from typing import Optional, Tuple

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Ch08 - ex08").getOrCreate()

collection = [1, "two", 3.0, ("four", 4), {"five", 5}]

sc = spark.sparkContext

collection_rdd = sc.parallelize(collection)  # promote list to RDD

print(collection_rdd)


# ex 8.1
cnt = collection_rdd.map(lambda x: 1).reduce(add)
print(cnt)

# ex 8.2
a_rdd = sc.parallelize([0, 1, None, [], 0.0])
a_rdd.filter(lambda x: x).collect().show(5)


# ex 8.3
def temp_to_temp(value: float, _from: str, to: str) -> Optional[float]:
    _from = str.upper(_from)
    to = str.upper(to)
    assert _from in ("C", "F", "K", "R")
    assert to in ("C", "F", "K", "R")
    assert _from != to

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

    convert = from_to[(_from, to)]
    return convert()


# ex 8.4


@F.udf(T.LongType())
def naive_udf(t: int) -> Optional[float]:
    if not isinstance(t, int):
        return None
    return t * 3.14159


# ex 8.5
# use existing spark session
# spark = SparkSession.builder.appName("Ch08 - ex8.5").getOrCreate()


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


# ex 8.5
@F.udf(SparkFrac)  # careful to use the correct type
def add_two_fractions(first: Frac, second: Frac) -> Optional[Frac]:
    """Add two fractions represented as a 2-tuple of integers as a float."""
    f_num, f_denom = first
    s_num, s_denom = second

    if f_denom and s_denom:
        result = Fraction(f_num, f_denom) + Fraction(s_num, s_denom)
        return result.numerator, result.denominator

    return None


frac_df.printSchema()

frac_df = frac_df.withColumn(
    "fraction_doubled", add_two_fractions("reduced_fraction", "reduced_fraction")
)

frac_df.select("fraction", "reduced_fraction", "fraction_doubled").distinct().show(
    5, False
)


# ex 8.6
@F.udf(SparkFrac)
def py_reduce_fraction_2(frac: Frac) -> Optional[Frac]:
    """Reduce a fraction represented as a 2-tuple of integers."""
    num, denom = frac

    MIN_VALUE = -pow(2, 63)
    MAX_VALUE = pow(2, 63) - 1

    if not denom:
        return None

    left, right = Fraction(num, denom).as_integer_ratio()

    if (num < MIN_VALUE or num > MAX_VALUE) or (denom < MIN_VALUE or denom > MAX_VALUE):
        return None

    return left, right


# No, it doesn't change the type annotation. The type can stay the same as we merely limited the range for the fraction
frac_df = frac_df.withColumn("reduced_fraction_2", py_reduce_fraction_2("fraction"))

frac_df.select("fraction", "reduced_fraction", "reduced_fraction_2").distinct().show(
    5, False
)
