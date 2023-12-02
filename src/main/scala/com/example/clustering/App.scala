package com.example.clustering
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import scala.collection.mutable.PriorityQueue
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.col
import org.apache.log4j.{Level, Logger}
import java.util.concurrent.Executors
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import java.io._
import org.apache.spark.HashPartitioner
import org.apache.spark.sql.{Row, DataFrame}
// Custom partitioner based on clusterId
class ClusterPartitioner(partitions: Int) extends org.apache.spark.Partitioner {
  override def numPartitions: Int = partitions
  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[(Long, Long, Int)]  // Assuming (videoId, noImg, clusterId) is the key
    k._3 % numPartitions
  }
}

object App {
  def compareFeatures(imageFeatures: Vector, queryFeatures: Vector): Double = {
    val gamma = 0.5
    val sum = imageFeatures.toArray.zip(queryFeatures.toArray).map { case (p1, p2) => math.pow(p2 - p1, 2) }.sum
    math.exp(-gamma / sum)
  }
  def dtwDistance(series1: Array[Vector], series2: Seq[Vector]): Double = {

    val n = series1.length
    val m = series2.length
    // DTW matrix initialization
    val dtwMatrix = Array.ofDim[Double](n + 1, m + 1)
    for (i <- 1 to n; j <- 1 to m) {
      val cost = Vectors.sqdist(
        series1(i - 1),
        series2(j - 1)
      ) // Euclidean distance between frames
      dtwMatrix(i)(j) = cost + math.min(
        dtwMatrix(i - 1)(j),
        math.min(dtwMatrix(i)(j - 1), dtwMatrix(i - 1)(j - 1))
      )
    }
    dtwMatrix(n)(m)
  }
  def videoDataRDD(row: Row): ((Long, Long, Int), Vector) = {
    val video_id = Option(row.getAs[String]("vId")).map(_.toLong).getOrElse(0L)
    val no_img = Option(row.getAs[Long]("fId")).getOrElse(0L)
    val clusterId = Option(row.getAs[Int]("prediction")).getOrElse(-1)
    val orig_feature = Option(row.getAs[Vector]("features")).getOrElse(Vectors.dense(Array.empty[Double]))
    ((video_id, no_img, clusterId), orig_feature)
  }

  def videoDataPartitionedRDD(clusteredData: DataFrame, numClusters: Int): RDD[((Long, Long, Int), Vector)] = {
    clusteredData
      .rdd
      .map(videoDataRDD) // 각 행을 처리하여 ((video_id, no_img, clusterId), orig_feature) 튜플로 변환
      .filter(_._1._3 != -1) // clusterId가 -1인 경우 필터링
      .partitionBy(new ClusterPartitioner(numClusters)) // numClusters를 기반으로 한 사용자 정의 Partitioner를 사용하여 파티션
      .cache() // 캐시하여 재사용
  }


  def processAndPrintResults(results: Seq[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]], queryFeatures: Array[org.apache.spark.ml.linalg.Vector], outputWriter: PrintWriter): Unit = {
    val flattenedResults = results.flatten
      .distinct
      .sortBy { case (_, _, featuresList) => dtwDistance(queryFeatures, featuresList) }
      .take(10)

    flattenedResults.foreach { case (videoId, noImgs, featuresList) =>
      outputWriter.println(s"($videoId, ${noImgs.mkString(",")})")
      val dtwdistance = dtwDistance(queryFeatures, featuresList)
      outputWriter.println(s"dtwDistance $dtwdistance")
    }
  }

  def getClosestClusterIds(queryFeature: Vector, broadcastCentroids: Broadcast[Array[Vector]]): Seq[Int] = {
    broadcastCentroids.value.zipWithIndex.map { case (center, idx) =>
      (idx, Vectors.sqdist(center, queryFeature))
    }.sortBy(_._2).take(10).map(_._1)
  }

  def filterRDDByClusterIds(partitionedRDD: RDD[((Long, Long, Int), Vector)], closestClusterIds: Seq[Int]): RDD[((Long, Long, Int), Vector)] = {
    partitionedRDD.filter { case ((_, _, clusterId), _) =>
      closestClusterIds.contains(clusterId)
    }
  }

  def broadcastSearchKeys(spark: SparkSession, topKGlobal: Seq[((Long, Long), Double, Vector)]): Broadcast[Map[(Long, Long), Double]] = {
    val searchKeys = topKGlobal.map { case ((videoId, noImg), similarity, _) => ((videoId, noImg), similarity) }.toMap
    spark.sparkContext.broadcast(searchKeys)
  }

  def generateVideoClips(partitionedVideoData: RDD[((Long, Long, Int), Vector)], broadcastedKeys: Broadcast[Map[(Long, Long), Double]]): RDD[(Long, Seq[Long], Seq[Vector])] = {
    partitionedVideoData.flatMap { case ((videoId, noImg, _), features) =>
      val range = (noImg - 2) to (noImg + 2)
      if (range.exists(i => broadcastedKeys.value.contains((videoId, i)))) {
        val noImgs = range.toSeq
        val featureList = Seq.fill(5)(features) // 5번 복제
        Some((videoId, noImgs, featureList))
      } else {
        None
      }
    }
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val spark = SparkSession.builder().master("yarn").appName("SimilaritySearch").getOrCreate()
    import spark.implicits._
    println(
          "=========================================================================================="
    )
    println(
          "                                    Data loading                                    "
    )
    println(
          "=========================================================================================="
    )
    val rawData: DataFrame = spark.read.parquet("/input/output2.parquet")

    val arrayToVector: Seq[Float] => Vector = (a: Seq[Float]) => Vectors.dense(a.map(_.toDouble).toArray)
    val arrayToVectorUDF = udf(arrayToVector)
    val featureData = rawData.withColumn("features", arrayToVectorUDF(col("subGFeature")))
    // Perform K-means clustering to get 1000 clusters
    
    
    println(
          "=========================================================================================="
    )
    println(
          "                                    색인 생성                                    "
    )
    println(
          "=========================================================================================="
    )
    val numClusters = 10
    val kmeans = new KMeans().setK(numClusters).setSeed(1L)
    // val model = kmeans.fit(featureData)
    val model = KMeansModel.load("/input/km_model/")
    val clusteredData = model.transform(featureData).cache()
    clusteredData.count()
    val centroids = model.clusterCenters
    val broadcastCentroids = spark.sparkContext.broadcast(centroids)
    
    val partitionedRDD = videoDataPartitionedRDD(clusteredData, numClusters)
    partitionedRDD.count()
    val hashPartitionedVideoData = partitionedRDD.partitionBy(new HashPartitioner(100)).cache()
    partitionedRDD.count() // 파티션 개수는 예시

    println(
            "=========================================================================================="
    )
    println(
          "                                    warmup start                                    "
    )
    println(
          "=========================================================================================="
    )
    for (_ <- 1 to 10) {
      val k = 10
      val queryFeatures =  Array(
            Vectors.dense(Array.fill(32)(math.random())),
            Vectors.dense(Array.fill(32)(math.random())),
            Vectors.dense(Array.fill(32)(math.random()))
          )
      val futures: Seq[Future[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]]] = queryFeatures.zipWithIndex.map { case (queryFeature, i) =>
    
      Future {
        val closestClusterIds = getClosestClusterIds(queryFeature, broadcastCentroids)
        val filteredRDD = filterRDDByClusterIds(partitionedRDD, closestClusterIds)
        val topKPartitions = filteredRDD
          .mapPartitions(iter => topKIterator(iter, k, queryFeature), preservesPartitioning = true)
        val topKGlobal = topKPartitions
          .reduce((a, b) => (a ++ b).sortBy(-_._2).take(k))
        val broadcastedKeys = broadcastSearchKeys(spark, topKGlobal)
        
        val videoClips = generateVideoClips(hashPartitionedVideoData, broadcastedKeys)
        val result: Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])] = videoClips.take(10).toSeq
        result
      }(ExecutionContext.global)
      }
      // val results = performAsyncTasks(queryFeatures, k)
      val aggregatedResults: Future[Seq[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]]] = Future.sequence(futures)
      val result: Seq[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]] = Await.result(aggregatedResults, Duration.Inf)


      }

    
    println(
            "=========================================================================================="
    )
    println(
          "                                    warmup finish                                   "
    )
    println(
          "=========================================================================================="
    )


    val k = 10
    val iterations = 1000
    
    var totalTime: Double = 0
    val outputFile = new File("output.txt")
    val outputWriter = new PrintWriter(new BufferedWriter(new FileWriter(outputFile)))
    println(
          "=========================================================================================="
    )
    println(
          "                                    질의처리 시작                                    "
    )
    println(
          "=========================================================================================="
    )
    for (i <- 1 to iterations) {
    
    val queryFeatures = if (i % 100 == 0) {
        // Regenerate queryFeatures every 100 iterations
        Array(
          Vectors.dense(Array.fill(32)(math.random())),
          Vectors.dense(Array.fill(32)(math.random())),
          Vectors.dense(Array.fill(32)(math.random()))
        )
      } else {
        // Use the existing queryFeatures
        Array(
        Vectors.dense(Array.fill(32)(math.random())),
        Vectors.dense(Array.fill(32)(math.random())),
        Vectors.dense(Array.fill(32)(math.random()))
      )
      }
      
    val startTime = System.nanoTime()
    val futures: Seq[Future[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]]] = queryFeatures.zipWithIndex.map { case (queryFeature, i) =>
  
    Future {
      // val startTime = System.nanoTime()
      
      val closestClusterIds = getClosestClusterIds(queryFeature, broadcastCentroids)
      val filteredRDD = filterRDDByClusterIds(partitionedRDD, closestClusterIds)
      val topKPartitions = filteredRDD
        .mapPartitions(iter => topKIterator(iter, k, queryFeature), preservesPartitioning = true)
      val topKGlobal = topKPartitions
        .reduce((a, b) => (a ++ b).sortBy(-_._2).take(k))
      val broadcastedKeys = broadcastSearchKeys(spark, topKGlobal)
      
      val videoClips = generateVideoClips(partitionedVideoData, broadcastedKeys)
      val result: Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])] = videoClips.take(10).toSeq
      result
    }(ExecutionContext.global)
    }
    // val results = performAsyncTasks(queryFeatures, k)
    val aggregatedResults: Future[Seq[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]]] = Future.sequence(futures)
    val result: Seq[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]] = Await.result(aggregatedResults, Duration.Inf)

    processAndPrintResults(result, queryFeatures, outputWriter)
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    println(s"$i Elapsed time: $elapsedTime seconds")
    outputWriter.println(s"$i Elapsed time: $elapsedTime seconds")
    totalTime += elapsedTime.toDouble
    
    
    }    
    
    val averageTime = totalTime / iterations
    println(s"averageTime: $averageTime seconds")
    outputWriter.println(s" averageTime: $averageTime seconds")
    outputWriter.close()
    println(
          "=========================================================================================="
    )
    println(
          "                                    질의처리 완료                                    "
    )
    println(
          "=========================================================================================="
    )
    
    spark.stop()
}

  def topKIterator(iter: Iterator[((Long, Long, Int), Vector)], k: Int, queryFeatures: Vector): Iterator[Seq[((Long, Long), Double, Vector)]] = {
    val pq = new PriorityQueue[((Long, Long), Double, Vector)]()(Ordering.by(_._2))
    iter.foreach { case ((videoId, noImg, _), origFeatures) =>
      val similarity = compareFeatures(origFeatures, queryFeatures)
      pq.enqueue(((videoId, noImg), similarity, origFeatures))
      if (pq.size > k) {
        pq.dequeue()
      }
    }
    Iterator(Seq(pq.toSeq: _*))
  }
  
}