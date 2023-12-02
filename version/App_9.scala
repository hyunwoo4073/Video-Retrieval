package com.example.clustering
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import scala.collection.mutable.PriorityQueue
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.col
// Custom partitioner based on clusterId
class ClusterPartitioner(partitions: Int) extends org.apache.spark.Partitioner {
  override def numPartitions: Int = partitions
  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[(Long, Long, Int)]  // Assuming (videoId, noImg, clusterId) is the key
    k._3 % numPartitions
  }
}
// val arrayToVector: Seq[Float] => Vector = (a: Seq[Float]) => Vectors.dense(a.map(_.toDouble).toArray)
// val arrayToVectorUDF = udf(arrayToVector)
object App {
  def compareFeatures(imageFeatures: Vector, queryFeaturesBroadcast: Broadcast[Vector]): Double = {
    val queryFeatures = queryFeaturesBroadcast.value
    val gamma = 0.5
    val sum = imageFeatures.toArray.zip(queryFeatures.toArray).map { case (p1, p2) => math.pow(p2 - p1, 2) }.sum
    math.exp(-gamma / sum)
  }
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("yarn").appName("SimilaritySearch").getOrCreate()
    import spark.implicits._
    val rawData: DataFrame = spark.read.parquet("/input/scenegraphFeature_GAT_edgeattrall_0-8.parquet")
    rawData.printSchema()
    // Rename or select the correct column that contains the feature vectors
    val arrayToVector: Seq[Float] => Vector = (a: Seq[Float]) => Vectors.dense(a.map(_.toDouble).toArray)
    val arrayToVectorUDF = udf(arrayToVector)
    val featureData = rawData.withColumn("features", arrayToVectorUDF(col("subGFeature")))
    // Perform K-means clustering to get 1000 clusters
    val numClusters = 1000
    val kmeans = new KMeans().setK(numClusters).setSeed(1L)
    val model = kmeans.fit(featureData)
    val clusteredData = model.transform(featureData)
    val centroids = model.clusterCenters
    clusteredData.printSchema()
    // Partition RDD based on clusterId
    val partitionedRDD: RDD[((Long, Long, Int), Vector)] = clusteredData.rdd.map { row =>
      val video_id = Option(row.getAs[String]("vId")).map(_.toLong).getOrElse(0L)
      val no_img = Option(row.getAs[Long]("fId")).getOrElse(0L)
      val clusterId = Option(row.getAs[Int]("prediction")).getOrElse(-1)
      val orig_feature = Option(row.getAs[Vector]("features")).getOrElse(Vectors.dense(Array.empty[Double]))
      ((video_id, no_img, clusterId), orig_feature)
    }
    .filter(_._1._3 != -1)
    .partitionBy(new ClusterPartitioner(numClusters))
    .cache()
    
    partitionedRDD.count()
    // set up a k-nn query
    import spark.implicits._
    val queryFeatures = Vectors.dense(Array.fill(32)(math.random()))
    val broadcastQueryFeatures = spark.sparkContext.broadcast(queryFeatures)
    def topKIterator(iter: Iterator[((Long, Long, Int), Vector)], k: Int): Iterator[Seq[((Long, Long), Double, Vector)]] = {
      val pq = new PriorityQueue[((Long, Long), Double, Vector)]()(Ordering.by(_._2))
      iter.foreach { case ((videoId, noImg, _), origFeatures) =>
        val similarity = compareFeatures(origFeatures, broadcastQueryFeatures)
        pq.enqueue(((videoId, noImg), similarity, origFeatures))
        if (pq.size > k) {
          pq.dequeue()
        }
      }
      Iterator(Seq(pq.toSeq: _*))
    }
    val k = 10
    // start to process the k-nn query
    val startTime = System.nanoTime()
    // get 2 closest centroids to the query features
    val closestClusterIds: Array[Int] = centroids.zipWithIndex.map { case (center, idx) =>
      (idx, Vectors.sqdist(center, queryFeatures))
    }.sortBy(_._2).take(2).map(_._1)
    val filteredRDD = partitionedRDD.filter { case ((_, _, clusterId), _) =>
      closestClusterIds.contains(clusterId)
    }
    val topKPartitions = filteredRDD
      .mapPartitions(iter => topKIterator(iter, k), preservesPartitioning = true)
    val topKGlobal = topKPartitions
      .reduce((a, b) => (a ++ b).sortBy(-_._2).take(k))
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    println(s"Elapsed time: $elapsedTime seconds")
    spark.stop()
  }
}