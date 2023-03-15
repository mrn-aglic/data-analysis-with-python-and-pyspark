# The RDD, too old-school?

With the advent of the data frame, which boasts better performance and a streamlined API for common data operations (select, filter, groupby, join), the RDD fell behind in terms of popularity. Is there room for the RDD in a modern PySpark program?
While the data frame is becoming more and more flexible as the Spark versions are released, the RDD still reigns in terms of flexibility. The RDD especially shines in two use cases:
- When you have an unordered collection of Python objects that can be pickled (which is how Python calls object serialization; see http://mng.bz/M2X7).
- When you have unordered _key_, _value_ pairs, like in a Python dictionary.

Both use cases are covered in this chapter. The data frame should be your structure of choice by default, but know that if you find it restrictive, the RDD is waiting for you.

Unlike the data frame, where most of our data manipulation tool kit revolved around columns, RDD revolves around objects: I think of an RDD as a bag of elements with no order or relationship to one another. Each element is independent of the other.
