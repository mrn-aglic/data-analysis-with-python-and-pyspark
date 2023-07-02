from typing import Optional

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("Ch12 - recipes ML model - are you a dessert?")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

food = spark.read.csv(
    "/opt/spark/data/recipes/epi_r.csv", inferSchema=True, header=True
)

print(food.count(), len(food.columns))

food.printSchema()


def sanitize_column_name(name):
    answer = name

    for i, j in ((" ", "_"), ("-", "_"), ("/", "_"), ("&", "and")):
        answer = answer.replace(i, j)

    return "".join(
        [char for char in answer if char.isalpha() or char.isdigit() or char == "_"]
    )


food = food.toDF(*[sanitize_column_name(name) for name in food.columns])

interesting_columns = ["clove"]

for x in food.columns:
    if x in interesting_columns:
        food.select(x).summary().show()

import pandas as pd

pd.set_option("display.max.rows", 1000)

is_binary = food.agg(
    *[(F.size(F.collect_set(x)) == 2).alias(x) for x in food.columns]
).toPandas()

print(is_binary.unstack())

food.agg(*[F.collect_set(x) for x in ("cakeweek", "wasteless")]).show(1, False)

food.where("cakeweek > 1.0 or wasteless > 1.0").select(
    "title", "rating", "wasteless", "cakeweek", food.columns[-1]
).show()

food = food.where(
    (F.col("cakeweek").isin([0.0, 1.0]) | F.col("cakeweek").isNull())
    & (F.col("wasteless").isin([0.0, 1.0]) | F.col("wasteless").isNull())
)

print(food.count(), len(food.columns))

IDENTIFIERS = ["title"]
CONTINUOUS_COLUMNS = [
    "rating",
    "calories",
    "protein",
    "fat",
    "sodium",
]

TARGET_COLUMN = ["dessert"]

BINARY_COLUMNS = [
    x
    for x in food.columns
    if x not in [*IDENTIFIERS, *CONTINUOUS_COLUMNS, *TARGET_COLUMN]
]

food = food.dropna(how="all", subset=[x for x in food.columns if x not in IDENTIFIERS])

food = food.dropna(subset=TARGET_COLUMN)

print(food.count(), len(food.columns))

food = food.fillna(0.0, subset=BINARY_COLUMNS)

print(food.where(F.col(BINARY_COLUMNS[0]).isNull()).count())  # => 0


@F.udf(T.BooleanType())
def is_a_number(value: Optional[str]) -> bool:
    if not value:
        return True

    try:
        _ = float(value)
    except ValueError:
        return False
    return True


food.where(~is_a_number(F.col("rating"))).select(*CONTINUOUS_COLUMNS).show()

for column in ["rating", "calories"]:
    food = food.where(is_a_number(F.col(column)))
    food = food.withColumn(column, F.col(column).cast(T.DoubleType()))

print(food.count(), len(food.columns))

food.select(*CONTINUOUS_COLUMNS).summary(
    "mean",
    "stddev",
    "min",
    "1%",
    "5%",
    "50%",
    "95%",
    "99%",
    "max",
).show()

maximum = {"calories": 3203.0, "protein": 173.0, "fat": 207.0, "sodium": 5661.0}

for k, v in maximum.items():
    food = food.withColumn(
        k, F.when(F.isnull(F.col(k)), F.col(k)).otherwise(F.least(F.col(k), F.lit(v)))
    )


inst_sum_of_binary_columns = [F.sum(F.col(x)).alias(x) for x in BINARY_COLUMNS]

sum_of_binary_columns = food.select(*inst_sum_of_binary_columns).head().asDict()

num_rows = food.count()

too_rare_features = [
    k for k, v in sum_of_binary_columns.items() if v < 10 or v > (num_rows - 10)
]

print(len(too_rare_features))

print(too_rare_features)

BINARY_COLUMNS = list(set(BINARY_COLUMNS) - set(too_rare_features))

print(BINARY_COLUMNS)

food = food.withColumn(
    "protein_ratio", F.col("protein") * 4 / F.col("calories")
).withColumn("fat_ratio", F.col("fat") * 9 / F.col("calories"))

food = food.fillna(0.0, subset=["protein_ratio", "fat_ratio"])

CONTINUOUS_COLUMNS += ["protein_ratio", "fat_ratio"]

# We use the VectorAssembler transformer on the food data frame to create a new column,
# continuous_features, that contains a Vector of all our continuous features.
# A transformer is a preconfigured object that, as its name indicates, transforms a data frame.
from pyspark.ml.feature import VectorAssembler

continuous_features = VectorAssembler(
    inputCols=CONTINUOUS_COLUMNS, outputCol="continuous_features"
)

vector_food = food.select(CONTINUOUS_COLUMNS)

for x in CONTINUOUS_COLUMNS:
    vector_food = vector_food.where(~F.isnull(F.col(x)))

print("Food count:")
print(food.count(), len(food.columns))
print("Vector food count:")
print(vector_food.count(), len(vector_food.columns))

vector_variable = continuous_features.transform(vector_food)

vector_variable.select("continuous_features").show(3, False)

vector_variable.select("continuous_features").printSchema()

from pyspark.ml.stat import Correlation

correlation = Correlation.corr(vector_variable, "continuous_features")

correlation.printSchema()

correlation_array = correlation.head()[0].toArray()

correlation_pd = pd.DataFrame(
    correlation_array, index=CONTINUOUS_COLUMNS, columns=CONTINUOUS_COLUMNS
)

print(correlation_pd.iloc[:, :4])

print(correlation_pd.iloc[:, 4:])


from pyspark.ml.feature import Imputer

OLD_COLS = ["calories", "protein", "fat", "sodium"]
NEW_COLS = ["calories_i", "protein_i", "fat_i", "sodium_i"]

imputer = Imputer(strategy="mean", inputCols=OLD_COLS, outputCols=NEW_COLS)

imputer_model = imputer.fit(food)

CONTINUOUS_COLUMNS = list(set(CONTINUOUS_COLUMNS) - set(OLD_COLS)) + NEW_COLS

food_imputed = imputer_model.transform(food)

food_imputed.where("calories is null").select("calories", "calories_i").show(5, False)

from pyspark.ml.feature import MinMaxScaler

CONTINUOUS_NB = [x for x in CONTINUOUS_COLUMNS if "ratio" not in x]

continuous_assembler = VectorAssembler(inputCols=CONTINUOUS_NB, outputCol="continuous")

food_features = continuous_assembler.transform(food_imputed)

continuous_scaler = MinMaxScaler(inputCol="continuous", outputCol="continuous_scaled")

food_features = continuous_scaler.fit(food_features).transform(food_features)

food_features.select("continuous_scaled").show(3, False)
