import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.ml import Estimator, Model, Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
    HasInputCol,
    HasInputCols,
    HasOutputCol,
    HasOutputCols,
)

# add ability to serialize and deserialize metadata of the transformer and estimator
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame


# pylint: disable=too-many-arguments
class ScalarNAFiller(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasInputCols,
    HasOutputCols,
    DefaultParamsReadable,
    DefaultParamsWritable,
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
            raise ValueError("Only one of `inputCol` and `inputCols` must be set.")

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


# our custom Params mixin
class _ExtremeValueCapperParams(
    HasInputCol, HasOutputCol, DefaultParamsWritable, DefaultParamsReadable
):
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
