import pyspark.sql.functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Ch11 - word counting").getOrCreate()

results = (
    spark.read.text("/opt/spark/data/gutenberg_books/*.txt")
    .select(F.split(F.col("value"), " ").alias("line"))
    .select(F.explode(F.col("line")).alias("word"))
    .select(F.lower(F.col("word")).alias("word"))
    .select(F.regexp_extract(F.col("word"), "[a-z']+", 0).alias("word"))
    .where(F.col("word") != "")
    .groupby(F.col("word"))
    .count()
)

# results.orderBy(F.col("count").desc()).show(10)
# results.orderBy("count", ascending=False).show(10)
results.orderBy(F.desc("count")).show(10)

# WholeStageCodegen is a stage where each operation happens on the same pass over
# the data. For our example, we have three:
# 1. Splitting the value
# 2. Filtering out the empty words, extracting the words, and pre-aggregating the
# word counts (like we saw in section 11.1.3) # pre-aggregate = groupby and count for each file separately
# before reshuffling
# 3. Aggregating the data into a final data frame
