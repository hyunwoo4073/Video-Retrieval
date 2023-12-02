// package com.example.clustering
// import org.apache.spark.broadcast.Broadcast
// import org.apache.spark.ml.clustering.KMeans
// import org.apache.spark.ml.clustering.KMeansModel
// import org.apache.spark.ml.linalg.Vector
// import org.apache.spark.ml.linalg.Vectors
// import org.apache.spark.sql.DataFrame
// import org.apache.spark.sql.SparkSession
// import org.apache.spark.rdd.RDD
// import scala.collection.mutable.PriorityQueue
// import org.apache.spark.sql.functions.udf
// import org.apache.spark.sql.functions.col
// import org.apache.log4j.{Level, Logger}
// import java.util.concurrent.Executors
// import scala.concurrent.duration.Duration
// import scala.concurrent.{Await, ExecutionContext, Future}
// import scala.concurrent.ExecutionContext.Implicits.global
// import java.io.PrintWriter
// // Custom partitioner based on clusterId
// class ClusterPartitioner(partitions: Int) extends org.apache.spark.Partitioner {
//   override def numPartitions: Int = partitions
//   override def getPartition(key: Any): Int = {
//     val k = key.asInstanceOf[(Long, Long, Int)]  // Assuming (videoId, noImg, clusterId) is the key
//     k._3 % numPartitions
//   }
// }
// // val arrayToVector: Seq[Float] => Vector = (a: Seq[Float]) => Vectors.dense(a.map(_.toDouble).toArray)
// // val arrayToVectorUDF = udf(arrayToVector)
// object App {
//   def compareFeatures(imageFeatures: Vector, queryFeatures: Vector): Double = {
//     val gamma = 0.5
//     val sum = imageFeatures.toArray.zip(queryFeatures.toArray).map { case (p1, p2) => math.pow(p2 - p1, 2) }.sum
//     math.exp(-gamma / sum)
//   }
//   def dtwDistance(series1: Array[Vector], series2: Seq[Vector]): Double = {

//     val n = series1.length
//     val m = series2.length
//     // DTW matrix initialization
//     val dtwMatrix = Array.ofDim[Double](n + 1, m + 1)
//     for (i <- 1 to n; j <- 1 to m) {
//       val cost = Vectors.sqdist(
//         series1(i - 1),
//         series2(j - 1)
//       ) // Euclidean distance between frames
//       dtwMatrix(i)(j) = cost + math.min(
//         dtwMatrix(i - 1)(j),
//         math.min(dtwMatrix(i)(j - 1), dtwMatrix(i - 1)(j - 1))
//       )
//     }
//     dtwMatrix(n)(m)
//   }



//   def main(args: Array[String]): Unit = {
//     Logger.getLogger("org").setLevel(Level.ERROR)
//     Logger.getLogger("akka").setLevel(Level.ERROR)
//     val spark = SparkSession.builder().master("yarn").appName("SimilaritySearch").getOrCreate()
//     import spark.implicits._
//     val rawData: DataFrame = spark.read.parquet("/input/scenegraphFeature_GAT_edgeattrall_0-8.parquet")
//     // rawData.printSchema()
//     // Rename or select the correct column that contains the feature vectors
//     val arrayToVector: Seq[Float] => Vector = (a: Seq[Float]) => Vectors.dense(a.map(_.toDouble).toArray)
//     val arrayToVectorUDF = udf(arrayToVector)
//     val featureData = rawData.withColumn("features", arrayToVectorUDF(col("subGFeature")))
//     // Perform K-means clustering to get 1000 clusters
//     val numClusters = 10
//     val kmeans = new KMeans().setK(numClusters).setSeed(1L)
//     // val model = kmeans.fit(featureData)
//     val model = KMeansModel.load("/input/km_model/")
//     val warmUpIterations = 5

//     val dummyOperation: Unit = {
//       for (_ <- 1 to warmUpIterations) {
//         val clusteredDataWarmUp = model.transform(featureData)
//       }
//     }

//     // 처음에 warm-up을 실행
//     dummyOperation
//     // model.write.overwrite().save("/input/km_model")
//     val clusteredData = model.transform(featureData)
//     // clusteredData.count()
//     val centroids = model.clusterCenters
//     val broadcastCentroids = spark.sparkContext.broadcast(centroids)
//     // clusteredData.printSchema()
//     // Partition RDD based on clusterId
//     val partitionedRDD: RDD[((Long, Long, Int), Vector)] = clusteredData.rdd.map { row =>
//       val video_id = Option(row.getAs[String]("vId")).map(_.toLong).getOrElse(0L)
//       val no_img = Option(row.getAs[Long]("fId")).getOrElse(0L)
//       val clusterId = Option(row.getAs[Int]("prediction")).getOrElse(-1)
//       val orig_feature = Option(row.getAs[Vector]("features")).getOrElse(Vectors.dense(Array.empty[Double]))
//       ((video_id, no_img, clusterId), orig_feature)
//     }
//     .filter(_._1._3 != -1)
//     .partitionBy(new ClusterPartitioner(numClusters))
//     // .partitionBy(new ClusterPartitioner(2000))
//     .cache()
//     partitionedRDD.count()
//     // set up a k-nn query
//     val queryFeatures = Array(
//       Vectors.dense(Array.fill(32)(math.random())),
//       Vectors.dense(Array.fill(32)(math.random())),
//       Vectors.dense(Array.fill(32)(math.random()))
//     )
//     val k = 10
//     val iterations = 10
//     var totalTime: Double = 0
//     val startTime = System.nanoTime()
//     for (i <- 1 to iterations) {
    
//     val futures: Seq[Future[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]]] = queryFeatures.zipWithIndex.map { case (queryFeature, i) =>
//     Future {
//       // val startTime = System.nanoTime()
//       // get 2 closest centroids to the query features
//       val closestClusterIds = broadcastCentroids.value.zipWithIndex.map { case (center, idx) =>
//         (idx, Vectors.sqdist(center, queryFeature))
//       }.sortBy(_._2).take(10).map(_._1)
//       val filteredRDD = partitionedRDD.filter { case ((_, _, clusterId), _) =>
//         closestClusterIds.contains(clusterId)
//       }
//       val topKPartitions = filteredRDD
//         .mapPartitions(iter => topKIterator(iter, k, queryFeature), preservesPartitioning = true)
//       val topKGlobal = topKPartitions
//         .reduce((a, b) => (a ++ b).sortBy(-_._2).take(k))
//       val endTime = System.nanoTime()
//       val searchKeys: Set[(Long, Long)] = topKGlobal.map{ case ((videoId, noImg), _, _) => (videoId, noImg) }.collect { case key => key }.toSet
//       val broadcastedKeys: Broadcast[Set[(Long, Long)]] = spark.sparkContext.broadcast(searchKeys)
//       val videoClips = partitionedRDD.flatMap { case ((videoId, noImg, _), features) =>
//       val range = (noImg - 2) to (noImg + 2)
//       if (range.exists(i => broadcastedKeys.value.contains((videoId, i)))) {
//         Some((videoId, noImg, features))
//       } else {
//         None
//       }
      
//     }
//       val result: Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])] = videoClips.groupBy(_._1).map { case (videoId, clips) =>
//       val noImgs = clips.map(_._2).toSeq
//       val featuresList = clips.map(_._3).toSeq
//       (videoId, noImgs, featuresList)
//     }.collect().toSeq


      
//       // val elapsedTime = (endTime - startTime) / 1e9
//       // println(elapsedTime)
//       result
//     }(ExecutionContext.global)
//     }
//     // Await.result(Future.sequence(futures), Duration.Inf)
    
//     // val aggregatedResult: Future[Seq[Unit]] = Future.sequence(futures)
//     // val result: Seq[Unit] = Await.result(aggregatedResult, Duration.Inf)
//     result.foreach(println)
//     val aggregatedResults: Future[Seq[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]]] = Future.sequence(futures)
//     val result: Seq[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]] = Await.result(aggregatedResults, Duration.Inf)

// // 결과 출력
//     result.foreach { case seqResult =>
//     seqResult.foreach { case (videoId, noImgs, featuresList) =>
//       println(s"($videoId, ${noImgs.mkString(",")})")
//       val dtwdistance = dtwDistance(queryFeatures, featuresList)
//       println(s"dtwDistance $dtwdistance")
//     // elapsedTime.foreach(println)
//     }
//   }
// }  
// // }
//     val endTime = System.nanoTime()
//     val elapsedTime = (endTime - startTime) / 1e9
//     val TotalTime = elapsedTime / iterations
//     println(s" Elapsed time: $TotalTime seconds")
    
//     // totalTime += elapsedTime.toDouble
//   // }
//     // Wait for all futures to complete
//     // val averageTime = totalTime / iterations
//     // println(s"average time: $averageTime seconds")
    
//     spark.stop()
// }

//   def topKIterator(iter: Iterator[((Long, Long, Int), Vector)], k: Int, queryFeatures: Vector): Iterator[Seq[((Long, Long), Double, Vector)]] = {
//     val pq = new PriorityQueue[((Long, Long), Double, Vector)]()(Ordering.by(_._2))
//     iter.foreach { case ((videoId, noImg, _), origFeatures) =>
//       val similarity = compareFeatures(origFeatures, queryFeatures)
//       pq.enqueue(((videoId, noImg), similarity, origFeatures))
//       if (pq.size > k) {
//         pq.dequeue()
//       }
//     }
//     Iterator(Seq(pq.toSeq: _*))
//   }
 
// }



// // package com.example.clustering
// // import org.apache.spark.broadcast.Broadcast
// // import org.apache.spark.ml.clustering.KMeans
// // import org.apache.spark.ml.clustering.KMeansModel
// // import org.apache.spark.ml.linalg.Vector
// // import org.apache.spark.ml.linalg.Vectors
// // import org.apache.spark.sql.DataFrame
// // import org.apache.spark.sql.SparkSession
// // import org.apache.spark.rdd.RDD
// // import scala.collection.mutable.PriorityQueue
// // import org.apache.spark.sql.functions.udf
// // import org.apache.spark.sql.functions.col
// // import org.apache.log4j.{Level, Logger}
// // import java.util.concurrent.Executors
// // import scala.concurrent.duration.Duration
// // import scala.concurrent.{Await, ExecutionContext, Future}
// // import scala.concurrent.ExecutionContext.Implicits.global
// // // Custom partitioner based on clusterId
// // class ClusterPartitioner(partitions: Int) extends org.apache.spark.Partitioner {
// //   override def numPartitions: Int = partitions
// //   override def getPartition(key: Any): Int = {
// //     val k = key.asInstanceOf[(Long, Long, Int)]  // Assuming (videoId, noImg, clusterId) is the key
// //     k._3 % numPartitions
// //   }
// // }
// // // val arrayToVector: Seq[Float] => Vector = (a: Seq[Float]) => Vectors.dense(a.map(_.toDouble).toArray)
// // // val arrayToVectorUDF = udf(arrayToVector)
// // object App {
// //   def compareFeatures(imageFeatures: Vector, queryFeatures: Vector): Double = {
// //     val gamma = 0.5
// //     val sum = imageFeatures.toArray.zip(queryFeatures.toArray).map { case (p1, p2) => math.pow(p2 - p1, 2) }.sum
// //     math.exp(-gamma / sum)
// //   }
// //   def dtwDistance(series1: Array[Vector], series2: Seq[Vector]): Double = {

// //     val n = series1.length
// //     val m = series2.length
// //     // DTW matrix initialization
// //     val dtwMatrix = Array.ofDim[Double](n + 1, m + 1)
// //     for (i <- 1 to n; j <- 1 to m) {
// //       val cost = Vectors.sqdist(
// //         series1(i - 1),
// //         series2(j - 1)
// //       ) // Euclidean distance between frames
// //       dtwMatrix(i)(j) = cost + math.min(
// //         dtwMatrix(i - 1)(j),
// //         math.min(dtwMatrix(i)(j - 1), dtwMatrix(i - 1)(j - 1))
// //       )
// //     }
// //     dtwMatrix(n)(m)
// //   }



// //   def main(args: Array[String]): Unit = {
// //     Logger.getLogger("org").setLevel(Level.ERROR)
// //     Logger.getLogger("akka").setLevel(Level.ERROR)
// //     val spark = SparkSession.builder().master("yarn").appName("SimilaritySearch").getOrCreate()
// //     import spark.implicits._
// //     val rawData: DataFrame = spark.read.parquet("/input/scenegraphFeature_GAT_edgeattrall_0-8.parquet")
// //     rawData.printSchema()
// //     // Rename or select the correct column that contains the feature vectors
// //     val arrayToVector: Seq[Float] => Vector = (a: Seq[Float]) => Vectors.dense(a.map(_.toDouble).toArray)
// //     val arrayToVectorUDF = udf(arrayToVector)
// //     val featureData = rawData.withColumn("features", arrayToVectorUDF(col("subGFeature")))
// //     // Perform K-means clustering to get 1000 clusters
// //     val numClusters = 1000
// //     val kmeans = new KMeans().setK(numClusters).setSeed(1L)
// //     // val model = kmeans.fit(featureData)
// //     val model = KMeansModel.load("/input/km_model/")
// //     // model.write.overwrite().save("/input/km_model")
// //     val clusteredData = model.transform(featureData).cache()
// //     clusteredData.count()
// //     val centroids = model.clusterCenters
// //     val broadcastCentroids = spark.sparkContext.broadcast(centroids)
// //     clusteredData.printSchema()
// //     // Partition RDD based on clusterId
// //     val partitionedRDD: RDD[((Long, Long, Int), Vector)] = clusteredData.rdd.map { row =>
// //       val video_id = Option(row.getAs[String]("vId")).map(_.toLong).getOrElse(0L)
// //       val no_img = Option(row.getAs[Long]("fId")).getOrElse(0L)
// //       val clusterId = Option(row.getAs[Int]("prediction")).getOrElse(-1)
// //       val orig_feature = Option(row.getAs[Vector]("features")).getOrElse(Vectors.dense(Array.empty[Double]))
// //       ((video_id, no_img, clusterId), orig_feature)
// //     }
// //     .filter(_._1._3 != -1)
// //     .partitionBy(new ClusterPartitioner(numClusters))
// //     // .partitionBy(new ClusterPartitioner(2000))
// //     .cache()
// //     partitionedRDD.count()
// //     // set up a k-nn query
// //     val queryFeatures = Array(
// //       Vectors.dense(Array.fill(32)(math.random())),
// //       Vectors.dense(Array.fill(32)(math.random())),
// //       Vectors.dense(Array.fill(32)(math.random()))
// //     )
// //     val k = 10
// //     val iterations = 10
// //     var totalTime: Double = 0
// //     for (i <- 1 to iterations) {
// //     val startTime = System.nanoTime()
// //     val futures: Seq[Future[Unit]] = queryFeatures.zipWithIndex.map { case (queryFeature, i) =>
// //       Future {
// //         val startTime = System.nanoTime()
// //         // get 2 closest centroids to the query features
// //         val closestClusterIds = broadcastCentroids.value.zipWithIndex.map { case (center, idx) =>
// //           (idx, Vectors.sqdist(center, queryFeature))
// //         }.sortBy(_._2).take(10).map(_._1)
// //         val filteredRDD = partitionedRDD.filter { case ((_, _, clusterId), _) =>
// //           closestClusterIds.contains(clusterId)
// //         }
// //         val topKPartitions = filteredRDD
// //           .mapPartitions(iter => topKIterator(iter, k, queryFeature), preservesPartitioning = true)
// //         val topKGlobal = topKPartitions
// //           .reduce((a, b) => (a ++ b).sortBy(-_._2).take(k))
// //         // val featureArray: Seq[org.apache.spark.ml.linalg.Vector] =  topKGlobal.map { case ((_, _), _, videoclip) =>
// //         //   videoclip
// //         // }
// //         val topKKeys = topKGlobal.map { case ((videoId, noImg), _, _) => (videoId, noImg) }
// //         topKKeys.foreach(println)
// //         val broadcasttopKKeys = spark.sparkContext.broadcast(topKKeys)
     
// //         var resultByTopKKeys = List[((Long, Long, Int), Seq[org.apache.spark.ml.linalg.Vector])]()
   
      
// //         broadcasttopKKeys.value.foreach {
// //           case (keyVideoId, keyNoImg) =>
// //             // 현재 topKKey에 해당하는 필터링된 값들의 Vector 추출
// //             val vectorsForKey = filteredRDD
// //               .filter {
// //                 case ((currentVideoId, noImg, _), _) =>
// //                   keyVideoId == currentVideoId && (keyNoImg >= noImg - 2 && keyNoImg <= noImg + 2)
// //               }
// //               .take(50)  // 원하는 갯수로 수정
// //               .map {
// //                 case (_, vector) => vector
// //               }
// //               .toSeq // 추가: 명시적으로 Seq로 변환

// //             // 필터링된 결과를 리스트에 추가
// //             resultByTopKKeys = resultByTopKKeys :+ ((keyVideoId, keyNoImg, 0), vectorsForKey)
// //         }

// //         // 결과 출력
// //         resultByTopKKeys.foreach {
// //           case ((videoId, noImg, idx), vectors) =>
            
// //             val dtwdistance = dtwDistance(queryFeatures, vectors)
            
// //             println(s"DTW Distance: $dtwdistance")
// //         }

        
// //         val endTime = System.nanoTime()
// //         // val elapsedTime = (endTime - startTime) / 1e9
// //         // println(s"$i Elapsed time: $elapsedTime seconds")
// //       }(ExecutionContext.global)
// //     }
// //     Await.result(Future.sequence(futures), Duration.Inf)
// //     val endTime = System.nanoTime()
// //     val elapsedTime = (endTime - startTime) / 1e9
// //     println(s"$i Elapsed time: $elapsedTime seconds")
// //     totalTime += elapsedTime.toDouble
// //   }
// //     // Wait for all futures to complete
// //     val averageTime = totalTime / iterations
// //     println(s"average time: $averageTime seconds")
    
// //     spark.stop()
// // }

// //   def topKIterator(iter: Iterator[((Long, Long, Int), Vector)], k: Int, queryFeatures: Vector): Iterator[Seq[((Long, Long), Double, Vector)]] = {
// //     val pq = new PriorityQueue[((Long, Long), Double, Vector)]()(Ordering.by(_._2))
// //     iter.foreach { case ((videoId, noImg, _), origFeatures) =>
// //       val similarity = compareFeatures(origFeatures, queryFeatures)
// //       pq.enqueue(((videoId, noImg), similarity, origFeatures))
// //       if (pq.size > k) {
// //         pq.dequeue()
// //       }
// //     }
// //     Iterator(Seq(pq.toSeq: _*))
// //   }
// // }



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
import java.io.PrintWriter
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



  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("yarn").appName("SimilaritySearch").getOrCreate()
    import spark.implicits._
    val rawData: DataFrame = spark.read.parquet("/input/scenegraphFeature_GAT_edgeattrall_0-8.parquet")
    // rawData.printSchema()
    // Rename or select the correct column that contains the feature vectors
    val arrayToVector: Seq[Float] => Vector = (a: Seq[Float]) => Vectors.dense(a.map(_.toDouble).toArray)
    val arrayToVectorUDF = udf(arrayToVector)
    val featureData = rawData.withColumn("features", arrayToVectorUDF(col("subGFeature")))
    // Perform K-means clustering to get 1000 clusters
    val numClusters = 10
    val kmeans = new KMeans().setK(numClusters).setSeed(1L)
    // val model = kmeans.fit(featureData)
    val model = KMeansModel.load("/input/km_model/")
    // model.write.overwrite().save("/input/km_model")
    val clusteredData = model.transform(featureData)
    // clusteredData.count()
    val centroids = model.clusterCenters
    val broadcastCentroids = spark.sparkContext.broadcast(centroids)
    // clusteredData.printSchema()
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
    // .partitionBy(new ClusterPartitioner(2000))
    .cache()
    partitionedRDD.count()
    // set up a k-nn query
    val queryFeatures = Array(
      Vectors.dense(Array.fill(32)(math.random())),
      Vectors.dense(Array.fill(32)(math.random())),
      Vectors.dense(Array.fill(32)(math.random()))
    )
    val k = 10
    val iterations = 10
    val startTime = System.nanoTime()
    var totalTime: Double = 0
    for (i <- 1 to iterations) {
    
    val futures: Seq[Future[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]]] = queryFeatures.zipWithIndex.map { case (queryFeature, i) =>
    Future {
      // val startTime = System.nanoTime()
      // get 2 closest centroids to the query features
      val closestClusterIds = broadcastCentroids.value.zipWithIndex.map { case (center, idx) =>
        (idx, Vectors.sqdist(center, queryFeature))
      }.sortBy(_._2).take(10).map(_._1)
      val filteredRDD = partitionedRDD.filter { case ((_, _, clusterId), _) =>
        closestClusterIds.contains(clusterId)
      }
      val topKPartitions = filteredRDD
        .mapPartitions(iter => topKIterator(iter, k, queryFeature), preservesPartitioning = true)
      val topKGlobal = topKPartitions
        .reduce((a, b) => (a ++ b).sortBy(-_._2).take(k))
      val searchKeys: Set[(Long, Long)] = topKGlobal.map{ case ((videoId, noImg), _, _) => (videoId, noImg) }.collect { case key => key }.toSet
      val broadcastedKeys: Broadcast[Set[(Long, Long)]] = spark.sparkContext.broadcast(searchKeys)
      val videoClips = partitionedRDD.flatMap { case ((videoId, noImg, _), features) =>
      val range = (noImg - 2) to (noImg + 2)
      if (range.exists(i => broadcastedKeys.value.contains((videoId, i)))) {
        Some((videoId, noImg, features))
      } else {
        None
      }
      
    }
      val result: Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])] = videoClips.groupBy(_._1).map { case (videoId, clips) =>
      val noImgs = clips.map(_._2).toSeq
      val featuresList = clips.map(_._3).toSeq
      (videoId, noImgs, featuresList)
    }.collect().toSeq


      // val endTime = System.nanoTime()
      // val elapsedTime = (endTime - startTime) / 1e9
      // println(s"$i Elapsed time: $elapsedTime seconds")
      result
    }(ExecutionContext.global)
    }
    // Await.result(Future.sequence(futures), Duration.Inf)
    
    // val aggregatedResult: Future[Seq[Unit]] = Future.sequence(futures)
    // val result: Seq[Unit] = Await.result(aggregatedResult, Duration.Inf)
    // result.foreach(println)
    val aggregatedResults: Future[Seq[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]]] = Future.sequence(futures)
    val result: Seq[Seq[(Long, Seq[Long], Seq[org.apache.spark.ml.linalg.Vector])]] = Await.result(aggregatedResults, Duration.Inf)

// 결과 출력
    result.foreach { case seqResult =>
    seqResult.foreach { case (videoId, noImgs, featuresList) =>
      println(s"($videoId, ${noImgs.mkString(",")})")
      val dtwdistance = dtwDistance(queryFeatures, featuresList)
      println(s"dtwDistance $dtwdistance")
    }
  }
    }    
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    val TotalTime = elapsedTime / iterations
    println(s" Elapsed time: $TotalTime seconds")
    
    // totalTime += elapsedTime.toDouble
  // }
    // Wait for all futures to complete
    // val averageTime = totalTime / iterations
    // println(s"average time: $averageTime seconds")
    
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