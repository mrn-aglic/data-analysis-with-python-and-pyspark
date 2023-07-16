import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql import Column, DataFrame, SparkSession

spark = SparkSession.builder.appName(
    "Ch14 - Custom transformer ScalarNAFiller"
).getOrCreate()

test_df = spark.createDataFrame(
    [[1, 2, 4, 1], [3, 6, 5, 4], [9, 4, None, 9], [11, 17, None, 3]],
    ["one", "two", "three", "four"],
)


def scalarNAFillerFunction(
    df: DataFrame, inputCol: Column, outputCol: str, filler: float = 0.0
):
    return df.withColumn(outputCol, inputCol).fillna(filler, subset=outputCol)


scalarNAFillerFunction(test_df, F.col("three"), "five", -99.0).show()


class ScalarNAFiller(Transformer, HasInputCol, HasOutputCol):
    filler = Param(
        parent=Params._dummy(),  # to be consistent with other Params
        name="filler",
        doc="Value we want to replace our null value with",
        typeConverter=TypeConverters.toFloat,
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, filler=None):
        super().__init__()

        # We set the default value for the Param filler,
        # since keyword_only hijacks the regular default argument capture.
        self._setDefault(filler=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, filler=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setFiller(self, new_filler):
        return self.setParams(filler=new_filler)

    def getFiller(self):
        return self.getOrDefault(self.filler)

    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)

    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if not self.isSet("inputCol"):
            raise ValueError("No input column set for the ScalarNAFiller transformer")

        input_column = dataset[self.getInputCol()]
        output_column = self.getOutputCol()
        na_filler = self.getFiller()

        return dataset.withColumn(output_column, input_column.cast("double")).fillna(
            na_filler, output_column
        )


test_ScalarNAFiller = ScalarNAFiller(inputCol="three", outputCol="five", filler=-99)

test_ScalarNAFiller.transform(test_df).show()

# modify filler in place
test_ScalarNAFiller.setFiller(17).transform(test_df).show()

# temporarily override filler
test_ScalarNAFiller.transform(test_df, params={test_ScalarNAFiller.filler: 17}).show()
