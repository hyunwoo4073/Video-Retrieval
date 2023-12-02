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
    val rawData: DataFrame = spark.read.parquet("/input/scenegraphFeature_GAT_edgeattrall_0-8.parquet").repartition(60)
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
      .repartition(60)
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
    spark.stop()
  }
}

