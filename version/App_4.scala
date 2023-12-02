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
  //   val partitionSizes1 = defaultImgFeatureRDD.mapPartitions { iter =>
  // // Define a function to calculate the size of a single element
  //     def calculateSize(data: (Long, org.apache.spark.ml.linalg.Vector, org.apache.spark.ml.linalg.Vector, Long)): Long = {
  //       val size = 
  //         data._1.toString.getBytes("UTF-8").length.toLong +
  //         data._4.toString.getBytes("UTF-8").length.toLong +
  //         data._2.toDense.toArray.map(_.toString.getBytes("UTF-8").length.toLong).sum +
  //         data._3.toDense.toArray.map(_.toString.getBytes("UTF-8").length.toLong).sum
  //       size
  //     }

  //     // Calculate the size for each element in the partition
  //     val sizes = iter.map(row => calculateSize(row))
  //     Iterator(sizes.sum)
  //   }.collect()
  //   val partitionSizes2 = approxFeatureRDD.mapPartitions { iter =>
  // // Define a function to calculate the size of a single element
  //     def calculateSize(data: (Long, org.apache.spark.ml.linalg.Vector, org.apache.spark.ml.linalg.Vector, Long)): Long = {
  //       val size = 
  //         data._1.toString.getBytes("UTF-8").length.toLong +
  //         data._4.toString.getBytes("UTF-8").length.toLong +
  //         data._2.toDense.toArray.map(_.toString.getBytes("UTF-8").length.toLong).sum +
  //         data._3.toDense.toArray.map(_.toString.getBytes("UTF-8").length.toLong).sum
  //       size
  //     }

  //     // Calculate the size for each element in the partition
  //     val sizes = iter.map(row => calculateSize(row))
  //     Iterator(sizes.sum)
  //   }.collect()

  //   // Print partition sizes
  //   partitionSizes1.zipWithIndex.foreach { case (size, index) =>
  //     println(s"Partition $index Size: $size bytes")
  //   }
  //   // Print partition sizes
  //   partitionSizes2.zipWithIndex.foreach { case (size, index) =>
  //     println(s"Partition $index Size: $size bytes")
  //   }
    val partitionSizes = topKImagesRDD.mapPartitions { iter =>
      val sizes = iter.map { case (videoId, similarity, origFeatures, noImg, approxFeature) =>
        // Calculate the size of each element in the partition
        val size =
          videoId.toString.getBytes("UTF-8").length.toLong +
          similarity.toString.getBytes("UTF-8").length.toLong +
          // Calculate the size of origFeatures and approxFeature as needed
          // origFeatures.toDense.toArray.map(_.toString.getBytes("UTF-8").length.toLong).sum +
          // approxFeature.toDense.toArray.map(_.toString.getBytes("UTF-8").length.toLong).sum +
          noImg.toString.getBytes("UTF-8").length.toLong
        size
      }
      Iterator(sizes.sum)
    }.collect()

    // Print partition sizes
    partitionSizes.zipWithIndex.foreach { case (size, index) =>
      println(s"Partition $index Size: $size bytes")
    }
    // val partitionSize1 = defaultImgFeatureRDD.mapPartitions{it => Iterator(it.toSeq.size)}.collect.toSeq
    // val partitionSize2 = approxFeatureRDD.mapPartitions{it => Iterator(it.toSeq.size)}.collect.toSeq
    val partitionCount1 = defaultImgFeatureRDD.getNumPartitions
    val partitionCount2 = approxFeatureRDD.getNumPartitions
    val partitionCount3 = topKImagesRDD.getNumPartitions
    // val partitionCount4 = topKImages.getNumPartitions
    // println(s"defaultImgFeatureRDD partitionSize: $partitionSize1 ")
    // println(s"approxFeatureRDD partitionSize: $partitionSize2 ")
    println(s"defaultImgFeatureRDD partitionCount: $partitionCount1 ")
    println(s"approxFeatureRDD partitionCount: $partitionCount2 ")
    println(s"topKImagesRDD partitionCount: $partitionCount3 ")
    // println(s"topKImages partitionCount: $partitionCount4 ")

    spark.stop()
    }
  }

