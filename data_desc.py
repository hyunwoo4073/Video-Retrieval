from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("Data") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
df = spark.read.parquet("/input/output2.parquet")

print(
      "=========================================================================================="
    )
print(
      "                                    Data Schema                                     "
    )
print(
      "=========================================================================================="
    )
df.printSchema()
print("\n")
print(
      "=========================================================================================="
    )
print(
      "                                    Test Data                                     "
    )
print(
      "=========================================================================================="
    )
df.show(10, False)
print("\n")
print(
      "=========================================================================================="
    )
print(
      "                                    Total Row                                     "
    )
print(
      "=========================================================================================="
    )
row = df.count()
print(f"Total Row {row}")