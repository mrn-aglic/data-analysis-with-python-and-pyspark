from typing import Optional

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Imputer, MinMaxScaler, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession

# create spark session
spark = (
    SparkSession.builder.appName(
        "Ch13 - recipes ML model - are you a dessert? Transformers and estimators"
    )
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

# load data
food = spark.read.csv(
    "/opt/spark/data/recipes/epi_r.csv", inferSchema=True, header=True
)


def sanitize_column_name(name):
    answer = name

    for i, j in ((" ", "_"), ("-", "_"), ("/", "_"), ("&", "and")):
        answer = answer.replace(i, j)

    return "".join(
        [char for char in answer if char.isalpha() or char.isdigit() or char == "_"]
    )


# sanitize column names
food = food.toDF(*[sanitize_column_name(name) for name in food.columns])

# Check which columns contain binary data
is_binary = food.agg(
    *[(F.size(F.collect_set(x)) == 2).alias(x) for x in food.columns]
).toPandas()

# Remove rows that are illogical
food = food.where(
    (F.col("cakeweek").isin([0.0, 1.0]) | F.col("cakeweek").isNull())
    & (F.col("wasteless").isin([0.0, 1.0]) | F.col("wasteless").isNull())
)

# prepare different types of columns
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

# Drop rows where all columns contain only null values (ignoring title column)
food = food.dropna(how="all", subset=[x for x in food.columns if x not in IDENTIFIERS])

# drop the rows where the target column is null
food = food.dropna(subset=TARGET_COLUMN)

# assume that null values in binary columns are False
food = food.fillna(0.0, subset=BINARY_COLUMNS)


@F.udf(T.BooleanType())
def is_a_number(value: Optional[str]) -> bool:
    if not value:
        return True

    try:
        _ = float(value)
    except ValueError:
        return False
    return True


# cast the columns rating and calories into Double type
for column in ["rating", "calories"]:
    food = food.where(
        is_a_number(F.col(column))
    )  # remove the column if it's not a number
    food = food.withColumn(column, F.col(column).cast(T.DoubleType()))


maximum = {"calories": 3203.0, "protein": 173.0, "fat": 207.0, "sodium": 5661.0}

# limit the column to the 99th percentile, but keep null values as is
for k, v in maximum.items():
    food = food.withColumn(
        k, F.when(F.isnull(F.col(k)), F.col(k)).otherwise(F.least(F.col(k), F.lit(v)))
    )


# removing the binary columns that are heavily skewed towards one value (0 or 1)
inst_sum_of_binary_columns = [F.sum(F.col(x)).alias(x) for x in BINARY_COLUMNS]

sum_of_binary_columns = food.select(*inst_sum_of_binary_columns).head().asDict()

num_rows = food.count()

too_rare_features = [
    k for k, v in sum_of_binary_columns.items() if v < 10 or v > (num_rows - 10)
]

BINARY_COLUMNS = list(set(BINARY_COLUMNS) - set(too_rare_features))

# creating new features
food = food.withColumn(
    "protein_ratio", F.col("protein") * 4 / F.col("calories")
).withColumn("fat_ratio", F.col("fat") * 9 / F.col("calories"))

food = food.fillna(0.0, subset=["protein_ratio", "fat_ratio"])

CONTINUOUS_COLUMNS += ["protein_ratio", "fat_ratio"]

# We use the VectorAssembler transformer on the food data frame to create a new column,
# continuous_features, that contains a Vector of all our continuous features.
# A transformer is a preconfigured object that, as its name indicates, transforms a data frame.

# features columns into a single Vector column
continuous_features = VectorAssembler(
    inputCols=CONTINUOUS_COLUMNS, outputCol="continuous_features"
)

vector_food = food.select(CONTINUOUS_COLUMNS)

for x in CONTINUOUS_COLUMNS:
    vector_food = vector_food.where(~F.isnull(F.col(x)))


vector_variable = continuous_features.transform(vector_food)


# Getting the correlation matrix
correlation = Correlation.corr(vector_variable, "continuous_features")

correlation_array = correlation.head()[0].toArray()

correlation_pd = pd.DataFrame(
    correlation_array, index=CONTINUOUS_COLUMNS, columns=CONTINUOUS_COLUMNS
)


OLD_COLS = ["calories", "protein", "fat", "sodium"]
NEW_COLS = ["calories_i", "protein_i", "fat_i", "sodium_i"]

# taking care of filling null values with the mean value
imputer = Imputer(strategy="mean", inputCols=OLD_COLS, outputCols=NEW_COLS)

imputer_model = imputer.fit(food)

CONTINUOUS_COLUMNS = list(set(CONTINUOUS_COLUMNS) - set(OLD_COLS)) + NEW_COLS

food_imputed = imputer_model.transform(food)

# taking care to scaling the features to values between 0 and 1
CONTINUOUS_NB = ["rating", "calories_i", "protein_i", "fat_i", "sodium_i"]

continuous_assembler = VectorAssembler(inputCols=CONTINUOUS_NB, outputCol="continuous")

continuous_scaler = MinMaxScaler(inputCol="continuous", outputCol="continuous_scaled")
