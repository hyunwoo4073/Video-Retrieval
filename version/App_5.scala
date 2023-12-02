package com.example.clustering

/** @author
  *   ${user.name}
  */
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}
import scala.collection.mutable.PriorityQueue
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
    val sum = imageFeatures.toArray.zip(queryFeatures.toArray).map { case (p1, p2) => math.pow(p2 - p1, 2) }.sum
    math.exp(-gamma / sum)
  }
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("yarn").appName("SimilaritySearch").getOrCreate()
    val rawData: DataFrame = spark.read.parquet("/input/scenegraphFeature_GAT_edgeattrall_0-8.parquet")
    // Logger.getLogger("org").setLevel(Level.ERROR)
    // Logger.getLogger("akka").setLevel(Level.ERROR)
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
    val approxFeatureRDD = defaultImgFeatureRDD.map {
      case (videoId, origFeatures, approxFeatures, noImg) =>
        (videoId, origFeatures, approxFeatures, noImg)
    }.cache()
    approxFeatureRDD.count()
    defaultImgFeatureRDD.count()
    val startTime = System.nanoTime()
    def topKIterator(iter: Iterator[(Long, Vector, Vector, Long)], k: Int): Iterator[Seq[(Long, Double, Vector, Long, Vector)]] = {
      val pq = new PriorityQueue[(Long, Double, Vector, Long, Vector)]()(Ordering.by(_._2))
      iter.foreach { case (videoId, origFeatures, approxFeature, noImg) =>
        val similarity = compareFeatures(approxFeature, broadcastQueryFeatures)
        pq.enqueue((videoId, similarity, origFeatures, noImg, approxFeature))
        if (pq.size > k) {
          pq.dequeue()
        }
      }
      Iterator(Seq(pq.toSeq: _*))
    }
    val topKPartitions = defaultImgFeatureRDD
      .mapPartitions(iter => topKIterator(iter, k), preservesPartitioning = true)
    val topKGlobal = topKPartitions
      .reduce((a, b) => (a ++ b).sortBy(-_._2).take(k))
    val topKImagesRDD = spark.sparkContext.parallelize(topKGlobal)
    val topKImages = topKImagesRDD.toDF("video_Id", "similarity", "origFeatures", "no_img", "approxFeature")
    val numberOfTopKImages = topKImages.count()
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    println(s"Elapsed time: $elapsedTime seconds")
    spark.stop()
    }
  }

