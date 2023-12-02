package com.example.clustering
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import scala.collection.mutable.PriorityQueue
object App {
  def compareFeatures(imageFeatures: Vector, queryFeaturesBroadcast: Broadcast[Vector]): Double = {
    val queryFeatures = queryFeaturesBroadcast.value
    val gamma = 0.5
    val sum = imageFeatures.toArray.zip(queryFeatures.toArray).map { case (p1, p2) => math.pow(p2 - p1, 2) }.sum
    math.exp(-gamma / sum)
  }
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("yarn").appName("SimilaritySearch").getOrCreate()
    val rawData: DataFrame = spark.read.parquet("/input/scenegraphFeature_GAT_edgeattrall_0-8.parquet").repartition(90)
    import spark.implicits._
    val queryFeatures = Vectors.dense(Array.fill(64)(math.random()))
    val broadcastQueryFeatures = spark.sparkContext.broadcast(queryFeatures)
    val k = 10
    val defaultImgFeatureRDD = rawData.as[(String, Long, Array[Double])]
      .map { case (video_id, no_img, orig_feature) =>
        val origFeatures = Vectors.dense(orig_feature)
        ((video_id.toLong, no_img), origFeatures)
      }
      .rdd
      .repartition(180)
      .cache()
    defaultImgFeatureRDD.count()
    val startTime = System.nanoTime()
    def topKIterator(iter: Iterator[((Long, Long), Vector)], k: Int): Iterator[Seq[((Long, Long), Double, Vector)]] = {
      val pq = new PriorityQueue[((Long, Long), Double, Vector)]()(Ordering.by(_._2))
      iter.foreach { case ((videoId, noImg), origFeatures) =>
        val similarity = compareFeatures(origFeatures, broadcastQueryFeatures)
        pq.enqueue(((videoId, noImg), similarity, origFeatures))
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
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    println(s"Elapsed time: $elapsedTime seconds")
    
    
  //   val partitionSizes = defaultImgFeatureRDD.mapPartitions { iter =>
  // // Define a function to calculate the size of a single element
  //     def calculateSize(data: ((Long, Long), org.apache.spark.ml.linalg.Vector)): Long = {
  //       val size = 
  //         data._1._1.toString.getBytes("UTF-8").length.toLong +
  //         data._1._2.toString.getBytes("UTF-8").length.toLong +
  //         data._2.toDense.toArray.map(_.toString.getBytes("UTF-8").length.toLong).sum
  //       size
  //     }

  //     // Calculate the size for each element in the partition
  //     val sizes = iter.map(row => calculateSize(row))
  //     Iterator(sizes.sum)
  //   }.collect()

    // Print partition sizes
    // partitionSizes.zipWithIndex.foreach { case (size, index) =>
    //   println(s"Partition $index Size: $size bytes")
    // }

    val allVectors = defaultImgFeatureRDD.map { case (_, origFeatures) => origFeatures }

    // "가장 큰 값" 및 "가장 작은 값" 초기화
    var maxScalar = Double.MinValue
    var minScalar = Double.MaxValue

    // RDD를 순회하면서 "가장 큰 값" 및 "가장 작은 값" 갱신
    allVectors.collect().foreach { vector =>
      vector.toArray.foreach { value =>
        maxScalar = Math.max(maxScalar, value)
        minScalar = Math.min(minScalar, value)
      }
    }

    // "가장 큰 값"과 "가장 작은 값" 출력
    println("Max Scalar: " + maxScalar)
    println("Min Scalar: " + minScalar)

    spark.stop()
  }
}

