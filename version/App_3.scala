package com.example.clustering

/** @author
  *   ${user.name}
  */
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.{DoubleType, LongType, StructField, StructType}
import scala.util.Random
object App {
  def ax_feature(features: Vector, numBuckets: Int): Vector = {
    val bucketSize = 1.0 / numBuckets
    Vectors.dense(features.toArray.map { value =>
      val bucketIndex = math.min((value / bucketSize).toInt, numBuckets - 1)
      if (value >= 0.0 && value <= 1.0) bucketIndex.toDouble else value
    })
  }
  def compareFeatures(imageFeatures: Vector, queryFeaturesBroadcast: Broadcast[Vector]): Double = {
    val queryFeatures = queryFeaturesBroadcast.value
    val gamma = 0.5
    val sum = imageFeatures.toArray
              .zip(queryFeatures.toArray)
              .map { case (p1, p2) =>
                  math.pow(p2 - p1, 2)
              }.sum
    math.exp(-gamma / sum)
  }
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("yarn").appName("SimilaritySearch").config("spark.serializer", "org.apache.spark.serializer.KryoSerializer").getOrCreate()
    val rawData: DataFrame = spark.read.parquet("/input/output2.parquet")
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    import spark.implicits._
    val queryFeatures = Vectors.dense(Array.fill(64)(math.random()))
    val broadcastQueryFeatures = spark.sparkContext.broadcast(queryFeatures)
    val k = 10
    val defaultImgFeatureRDD = rawData.as[(String, Long, Array[Double])]
      .map { case (video_id, no_img, orig_feature) =>
        val origFeatures = Vectors.dense(orig_feature)
        val approxFeatures = ax_feature(origFeatures, 5)
        (video_id.toLong, origFeatures, approxFeatures, no_img)
      }
      .rdd
      .cache()
    defaultImgFeatureRDD.count()
    val iterations = 100
    var totalTime: Double = 0
    for (i <- 1 to iterations) {
    val startTime = System.nanoTime()
    
    // println(s"startTime time: $startTime seconds")
    val similarities = defaultImgFeatureRDD.flatMap { case (videoId, origFeatures, approxFeature, noImg) =>
      val similarity = compareFeatures(approxFeature, broadcastQueryFeatures)
      Seq((videoId, similarity, origFeatures, noImg, approxFeature))
    }
    val topKImages = similarities.toDF("video_Id", "similarity", "origFeatures", "no_img", "approxFeature")
      .orderBy($"similarity".desc)
      .limit(k)
    val numberOfTopKImages = topKImages.count()
    val endTime = System.nanoTime()
    // println(s"end time: $endTime seconds")
    val executionTime = (endTime - startTime) / 1e9
    totalTime += executionTime.toDouble
    // println(s"Execution time: $executionTime seconds")
    val averageTime = totalTime / iterations
    println(s"Execution time: $averageTime seconds")
  }
}
}
