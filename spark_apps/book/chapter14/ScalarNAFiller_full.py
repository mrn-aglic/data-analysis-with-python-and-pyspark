import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
    HasInputCol,
    HasInputCols,
    HasOutputCol,
    HasOutputCols,
)
from pyspark.sql import Column, DataFrame, SparkSession

spark = SparkSession.builder.appName(
    "Ch14 - Custom transformer ScalarNAFiller full"
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


class ScalarNAFiller(
    Transformer, HasInputCol, HasOutputCol, HasInputCols, HasOutputCols
):
    filler = Param(
        parent=Params._dummy(),  # to be consistent with other Params
        name="filler",
        doc="Value we want to replace our null value with",
        typeConverter=TypeConverters.toFloat,
    )

    @keyword_only
    def __init__(
        self,
        inputCol=None,
        outputCol=None,
        inputCols=None,
        outputCols=None,
        filler=None,
    ):
        super().__init__()

        # We set the default value for the Param filler,
        # since keyword_only hijacks the regular default argument capture.
        self._setDefault(filler=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        inputCol=None,
        outputCol=None,
        inputCols=None,
        outputCols=None,
        filler=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setFiller(self, new_filler):
        return self.setParams(filler=new_filler)

    def getFiller(self):
        return self.getOrDefault(self.filler)

    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)

    def setInputCols(self, new_inputCols):
        return self.setParams(inputCols=new_inputCols)

    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)

    def setOutputCols(self, new_outputCols):
        return self.setParams(outputCols=new_outputCols)

    def check_params(self):
        # Test #1: either inputCol or inputCols can be set (but not both).
        if self.isSet("inputCol") and (self.isSet("inputCols")):
            raise ValueError("Only one of `inputCol` and `inputCols`" "must be set.")

        # Test #2: at least one of inputCol or inputCols must be set.
        if not (self.isSet("inputCol") or self.isSet("inputCols")):
            raise ValueError("One of `inputCol` or `inputCols` must be set.")

        # Test #3: if `inputCols` is set, then `outputCols`
        # must be a list of the same len()
        if self.isSet("inputCols"):
            if len(self.getInputCols()) != len(self.getOutputCols()):
                raise ValueError(
                    "The length of `inputCols` does not match"
                    " the length of `outputCols`"
                )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        self.check_params()

        input_columns = (
            [self.getInputCol()] if self.isSet("inputCol") else self.getInputCols()
        )
        output_columns = (
            [self.getOutputCol()] if self.isSet("outputCol") else self.getOutputCols()
        )

        answer = dataset

        if input_columns != output_columns:
            for in_col, out_col in zip(input_columns, output_columns):
                answer = answer.withColumn(out_col, F.col(in_col))

        na_filler = self.getFiller()

        return answer.fillna(na_filler, output_columns)


test_ScalarNAFiller = ScalarNAFiller(inputCol="three", outputCol="five", filler=-99)

test_ScalarNAFiller.transform(test_df).show()

# modify filler in place
test_ScalarNAFiller.setFiller(17).transform(test_df).show()

# temporarily override filler
test_ScalarNAFiller.transform(test_df, params={test_ScalarNAFiller.filler: 17}).show()
