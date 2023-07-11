from typing import Optional

import pyspark.ml.feature as MF
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

# create spark session
spark = (
    SparkSession.builder.appName(
        "Ch13 - recipes ML model - are you a dessert? ML PIPELINE"
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


OLD_COLS = ["calories", "protein", "fat", "sodium"]
NEW_COLS = ["calories_i", "protein_i", "fat_i", "sodium_i"]

CONTINUOUS_NB = ["rating", "calories_i", "protein_i", "fat_i", "sodium_i"]


# taking care of filling null values with the mean value
imputer = MF.Imputer(strategy="mean", inputCols=OLD_COLS, outputCols=NEW_COLS)

continuous_assembler = MF.VectorAssembler(
    inputCols=CONTINUOUS_NB, outputCol="continuous"
)

# taking care to scaling the features to values between 0 and 1
continuous_scaler = MF.MinMaxScaler(
    inputCol="continuous", outputCol="continuous_scaled"
)

food_pipeline = Pipeline(stages=[imputer, continuous_assembler, continuous_scaler])

pre_ml_assembler = MF.VectorAssembler(
    inputCols=BINARY_COLUMNS + ["continuous_scaled", "protein_ratio", "fat_ratio"],
    outputCol="features",
)

food_pipeline.setStages(
    [imputer, continuous_assembler, continuous_scaler, pre_ml_assembler]
)

food_pipeline_model = food_pipeline.fit(food)
food_features = food_pipeline_model.transform(food)


# An array of nonzero values
#       Dense:  [0.0, 1.0, 4.0, 0.0]
#       Sparse: (4, [1,2], [1.0, 4.0]) - first list is the position of non-zero values
food_features.select("title", "dessert", "features").show(5, truncate=30)

print(food_features.schema["features"])

print(food_features.schema["features"].metadata)

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    featuresCol="features", labelCol="dessert", predictionCol="prediction"
)

food_pipeline.setStages(
    [imputer, continuous_assembler, continuous_scaler, pre_ml_assembler, lr]
)

train, test = food.randomSplit([0.7, 0.3], 13)  # 13 is the seed
train.cache()

food_pipeline_model = food_pipeline.fit(train)
results = food_pipeline_model.transform(test)

# evaluating the model - confusion matrix
results.groupby("dessert").pivot("prediction").count().show()

# evaluating the model - precision and recall
lr_model = food_pipeline_model.stages[-1]
metrics = lr_model.evaluate(results.select("title", "dessert", "features"))

print(f"Model precision: {metrics.precisionByLabel[1]}")
print(f"Model recall: {metrics.recallByLabel[1]}")

# evaluating the model - ROC curve
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(
    labelCol="dessert", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

accuracy = evaluator.evaluate(results)
print(f"Area under ROC = {accuracy}")

# optimizing hyperparameters with cross-validation
from pyspark.ml.tuning import ParamGridBuilder

grid_search = ParamGridBuilder().addGrid(lr.elasticNetParam, [0.0, 1.0]).build()

print(grid_search)

from pyspark.ml.tuning import CrossValidator

cv = CrossValidator(
    estimator=food_pipeline,
    estimatorParamMaps=grid_search,
    evaluator=evaluator,
    numFolds=3,
    seed=13,
    collectSubModels=True,
)

cv_model = cv.fit(train)

print(cv_model.avgMetrics)

pipeline_food_model = cv_model.bestModel


# extracting feature names from the features vector
import pandas as pd

feature_names = ["(intercept)"] + [
    x["name"]
    for x in food_features.schema["features"].metadata["ml_attr"]["attrs"]["numeric"]
]

feature_coefficients = [lr_model.intercept] + list(lr_model.coefficients.values)

coefficients = pd.DataFrame(feature_coefficients, index=feature_names, columns=["coef"])

coefficients["abs_coef"] = coefficients["coef"].abs()

print(coefficients.sort_values(["abs_coef"]))
