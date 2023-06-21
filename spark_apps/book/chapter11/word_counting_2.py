import pyspark.sql.functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Ch11 - word counting").getOrCreate()

# narrow vs wide operations
# narrow operations are indifferent on the location of the data
# narrow operations do not require any shuffling of the data (records)
# grouping data, calculating the maximum and joining tables are WIDE operations.
# wide operations require the data to be laid out in a certain way between the multiple nodes
# in the first (previous word_counting.py) example the groupby and count transformation was split into
# two stages. The two stages were separated by an exchange (shuffling) of the records across nodes.
# wide operations require us to send the data across the network and thus incur a perfromance cost.


# example of a poorly written program.
# 1. The filter can be combined with the previous filter (spark can combine this in the plan it generates)
# 2. The first groupby is unnecessary (spark CAN'T figure out this)
results = (
    spark.read.text("/opt/spark/data/gutenberg_books/*.txt")
    .select(F.split(F.col("value"), " ").alias("line"))
    .select(F.explode(F.col("line")).alias("word"))
    .select(F.lower(F.col("word")).alias("word"))
    .select(F.regexp_extract(F.col("word"), "[a-z']+", 0).alias("word"))
    .where(F.col("word") != "")
    .groupby(F.col("word"))
    .count()
    .where(F.length(F.col("word")) > 8)  # new line
    .groupby(F.length(F.col("word")))  # new line
    .sum("count")
)

# results.orderBy(F.desc("count")).show(10)
results.show(5, False)

results.explain("formatted")

# Improved over the previous
results_bis = (
    spark.read.text("/opt/spark/data/gutenberg_books/*.txt")
    .select(F.split(F.col("value"), " ").alias("line"))
    .select(F.explode(F.col("line")).alias("word"))
    .select(F.lower(F.col("word")).alias("word"))
    .select(F.regexp_extract(F.col("word"), "[a-z']+", 0).alias("word"))
    .where(F.col("word") != "")
    .where(F.length(F.col("word")) > 8)  # new line
    .groupby(F.length(F.col("word")))  # new line
    .count()
)

results_bis.show(5, False)

results_bis.explain("formatted")

# WholeStageCodegen is a stage where each operation happens on the same pass over
# the data. For our example, we have three:
# 1. Splitting the value
# 2. Filtering out the empty words, extracting the words, and pre-aggregating the
# word counts (like we saw in section 11.1.3) # pre-aggregate = groupby and count for each file separately
# before reshuffling
# 3. Aggregating the data into a final data frame
