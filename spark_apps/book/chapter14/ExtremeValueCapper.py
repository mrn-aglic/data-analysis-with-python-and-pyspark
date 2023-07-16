import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import Column, DataFrame, SparkSession

spark = SparkSession.builder.appName("Ch14 - Custom estimator").getOrCreate()

test_df = spark.createDataFrame(
    [[1, 2, 4, 1], [3, 6, 5, 4], [9, 4, None, 9], [11, 17, None, 3]],
    ["one", "two", "three", "four"],
)


# blueprint function
def test_ExtremeValueCapper_transform(
    df: DataFrame, inputCol: Column, outputCol: str, cap: float, floor: float
):
    return df.withColumn(
        outputCol,
        F.when(inputCol > cap, cap).when(inputCol < floor, floor).otherwise(inputCol),
    )


# blueprint function
def test_ExtremeValueCapper_fit(
    df: DataFrame, inputCol: Column, outputCol: str, boundary: float
):
    avg, stddev = df.agg(F.mean(inputCol), F.stddev(inputCol)).head()

    cap = avg + boundary * stddev
    floor = avg - boundary * stddev

    return test_ExtremeValueCapper_transform(df, inputCol, outputCol, cap, floor)


# our custom Params mixin
class _ExtremeValueCapperParams(HasInputCol, HasOutputCol):
    boundary = Param(
        parent=Params._dummy(),
        name="boundary",
        doc="Multiple of standard deviation for the cap and floor. Default = 0.0.",
        typeConverter=TypeConverters.toFloat,
    )

    # Because this Mixin will not be directly called
    # - the call will come from when we call super() in a class that inherits from our Mixin -
    # we need to accept the arguments of any downstream transformer, model, or estimator.
    # In Python, we simply do this by passing * args to our initializer.
    def __init__(self, *args):
        super().__init__(
            *args
        )  # Ensure proper superclass call hierarchy by capturing them all under *args
        self._setDefault(boundary=0.0)

    def getBoundary(self):
        return self.getOrDefault(self.boundary)


# this should be the output of the estimator fit method
# it is customary to propagate the Params of the estimator to the companion model,
# even if they are not used.
# In our case, it means that boundary will be added to the Params of ExtremeValueCapperModel.
class ExtremeValueCapperModel(Model, _ExtremeValueCapperParams):
    cap = Param(
        parent=Params._dummy(),
        name="cap",
        doc="Upper bound of the values inputCol can take. Value will be capped to this value",
        typeConverter=TypeConverters.toFloat,
    )

    floor = Param(
        parent=Params._dummy(),
        name="floor",
        doc="Lower bound of the values inputCol can take. Value will be floored to this value",
        typeConverter=TypeConverters.toFloat,
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, cap=None, floor=None):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, cap=None, floor=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setCap(self, new_cap):
        return self.setParams(cap=new_cap)

    def getCap(self):
        return self.getOrDefault("cap")

    def setFloor(self, new_floor):
        return self.setParams(floor=new_floor)

    def getFloor(self):
        return self.getOrDefault("floor")

    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)

    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        if not self.isSet("inputCol"):
            raise ValueError(
                "No input column set for the ExtremeValueCapperModel transformer"
            )

        input_column = dataset[self.getInputCol()]
        output_column = self.getOutputCol()

        cap_value = self.getOrDefault("cap")
        floor_value = self.getOrDefault("floor")

        return dataset.withColumn(
            output_column,
            F.when(input_column > cap_value, cap_value)
            .when(input_column < floor_value, floor_value)
            .otherwise(input_column),
        )


class ExtremeValueCapper(Estimator, _ExtremeValueCapperParams):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, boundary=None):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, boundary=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setBoundary(self, new_boundary):
        self.setParams(boundary=new_boundary)

    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)

    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)

    def _fit(self, dataset):
        input_column = self.getInputCol()
        output_column = self.getOutputCol()
        boundary = self.getBoundary()

        avg, stddev = dataset.agg(F.mean(input_column), F.stddev(input_column)).head()

        cap_value = avg + boundary * stddev
        floor_value = avg - boundary * stddev

        return ExtremeValueCapperModel(
            inputCol=input_column,
            outputCol=output_column,
            cap=cap_value,
            floor=floor_value,
        )


test_EVC = ExtremeValueCapper(inputCol="one", outputCol="five", boundary=1.0)

test_EVC.fit(test_df).transform(test_df).show()
