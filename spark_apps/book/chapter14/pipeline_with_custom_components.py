from typing import Optional

import pyspark.ml.feature as MF
import pyspark.sql.functions as F
import pyspark.sql.types as T
from classes import ExtremeValueCapper, ScalarNAFiller
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession

# create spark session
spark = (
    SparkSession.builder.appName(
        "Ch14 - recipes ML model - are you a dessert? ML PIPELINE WITH CUSTOM STAGES"
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
    if x not in CONTINUOUS_COLUMNS and x not in TARGET_COLUMN and x not in IDENTIFIERS
]

# Drop rows where all columns contain only null values (ignoring title column)
food = food.dropna(how="all", subset=[x for x in food.columns if x not in IDENTIFIERS])

# drop the rows where the target column is null
food = food.dropna(subset=TARGET_COLUMN)


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


# maximum = {"calories": 3203.0, "protein": 173.0, "fat": 207.0, "sodium": 5661.0}

# limit the column to the 99th percentile, but keep null values as is
# for k, v in maximum.items():
#     food = food.withColumn(
#         k, F.when(F.isnull(F.col(k)), F.col(k)).otherwise(F.least(F.col(k), F.lit(v)))
#     )


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

# food = food.fillna(0.0, subset=["protein_ratio", "fat_ratio"])

# taking care of filling null values with the mean value
imputer = MF.Imputer(
    strategy="mean",
    inputCols=["calories", "protein", "fat", "sodium", "protein_ratio", "fat_ratio"],
    outputCols=[
        "calories_i",
        "protein_i",
        "fat_i",
        "sodium_i",
        "protein_ratio_i",
        "fat_ratio_i",
    ],
)

continuous_assembler = MF.VectorAssembler(
    inputCols=["rating", "calories_i", "protein_i", "fat_i", "sodium_i"],
    outputCol="continuous",
)

# taking care to scaling the features to values between 0 and 1
continuous_scaler = MF.MinMaxScaler(
    inputCol="continuous", outputCol="continuous_scaled"
)


pre_ml_assembler = MF.VectorAssembler(
    inputCols=BINARY_COLUMNS + ["continuous_scaled", "protein_ratio_i", "fat_ratio_i"],
    outputCol="features",
)

# new things from Chapter 14.3
scalar_na_filler = ScalarNAFiller(
    inputCols=BINARY_COLUMNS, outputCols=BINARY_COLUMNS, filler=0.0
)

boundary = 2.0

extreme_value_capper_cal = ExtremeValueCapper(
    inputCol="calories", outputCol="calories", boundary=boundary
)

extreme_value_capper_pro = ExtremeValueCapper(
    inputCol="protein", outputCol="protein", boundary=boundary
)

extreme_value_capper_fat = ExtremeValueCapper(
    inputCol="fat", outputCol="fat", boundary=boundary
)

extreme_value_capper_sod = ExtremeValueCapper(
    inputCol="sodium", outputCol="sodium", boundary=boundary
)

lr = LogisticRegression(
    featuresCol="features", labelCol="dessert", predictionCol="prediction"
)

food_pipeline = Pipeline(
    stages=[
        scalar_na_filler,
        extreme_value_capper_cal,
        extreme_value_capper_pro,
        extreme_value_capper_fat,
        extreme_value_capper_sod,
        imputer,
        continuous_assembler,
        continuous_scaler,
        pre_ml_assembler,
        lr,
    ]
)

# transforming the pipeline on our training data set
train, test = food.randomSplit(weights=[0.7, 0.3], seed=13)

food_pipeline_model = food_pipeline.fit(train)

results = food_pipeline_model.transform(test)

evaluator = BinaryClassificationEvaluator(
    labelCol="dessert", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

accuracy = evaluator.evaluate(results)

print(f"Area under ROC = {accuracy}")

# saving and reading the model
from pyspark.ml.pipeline import PipelineModel

food_pipeline_model.save("data/code/food_pipeline.model")
food_pipeline_model = PipelineModel.read().load("data/code/food_pipeline.model")
