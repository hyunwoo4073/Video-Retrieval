// package com.example.clustering
// import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
// import org.apache.spark.ml.linalg.{Vector, Vectors}
// import org.apache.spark.sql.{DataFrame, SparkSession}
// import org.apache.log4j.{Level, Logger}
// import scala.collection.mutable.PriorityQueue
// object App {
//   def main(args: Array[String]): Unit = {
//     val spark = SparkSession.builder().master("yarn").appName("SimilaritySearchWithLSH").getOrCreate()
//     val rawData: DataFrame = spark.read.parquet("/input/scenegraphFeature_GAT_edgeattrall_0-8.parquet")
//     import spark.implicits._
//     val queryFeatures = Vectors.dense(Array.fill(32)(math.random()))
//     val defaultImgFeatureDF = rawData.as[(String, Long, Array[Double])]
//       .map { case (video_id, no_img, orig_feature) =>
//         val origFeatures = Vectors.dense(orig_feature)
//         (video_id, no_img, origFeatures)
//       }
//       .toDF("video_id", "no_img", "features")
//       .repartition(90)
//       .cache()
//     val brp = new BucketedRandomProjectionLSH()
//       .setBucketLength(2.0)
//       .setNumHashTables(3)
//       .setInputCol("features")
//       .setOutputCol("hashes")
//     val model = brp.fit(defaultImgFeatureDF)
//     // Transform data and cache it
//     val transformedData = model.transform(defaultImgFeatureDF).cache()
//     transformedData.count()
//     val k = 10
//     val startTime = System.nanoTime()
//     val topKResults = model.approxNearestNeighbors(transformedData, queryFeatures, k, "EuclideanDistance")
//     topKResults.show()
//     val endTime = System.nanoTime()
//     val elapsedTime = (endTime - startTime) / 1e9
//     println(s"Elapsed time: $elapsedTime seconds")
//     spark.stop()
//   }
// }



package com.example.clustering
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import scala.collection.mutable.PriorityQueue
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}
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
    // val sliceArray = udf((array: Seq[Double]) => array.take(4)) // 처음 4개 요소만 남김
    // val newData = rawData.limit(1000)
    // DataFrame에서 subGFeature 컬럼을 슬라이스하여 새로운 컬럼을 생성합니다.
    // val newData = rawData.select(
    //   col("vId"),
    //   col("fId"),
    //   sliceArray(col("subGFeature")).as("subGFeature")
    // )
    // 4차원만 남은 새로운 DataFrame을 사용할 수 있습니다.
    // newData.show(truncate = false)
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    import spark.implicits._
    val queryFeatures = Vectors.dense(Array.fill(32)(math.random()))
    val broadcastQueryFeatures = spark.sparkContext.broadcast(queryFeatures)
    val k = 10
    val defaultImgFeatureRDD = rawData.as[(String, Long, Array[Double])]
      .map { case (video_id, no_img, orig_feature) =>
        val origFeatures = Vectors.dense(orig_feature)
        ((video_id.toLong, no_img), origFeatures)
      }
      .rdd
      .repartition(90)
      .cache()
    defaultImgFeatureRDD.count()

    val iterations = 10
    var totalTime: Double = 0
    for (i <- 1 to iterations) {
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
      println(s"$i Elapsed time: $elapsedTime seconds")
      totalTime += elapsedTime.toDouble
      
      
    }
    val averageTime = totalTime / iterations
    println(s"average time: $averageTime seconds")
    spark.stop()
  }
  
  
}




