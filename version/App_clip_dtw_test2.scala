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
    rawData.printSchema()
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
    val clusteredData = model.transform(featureData).cache()
    clusteredData.count()
    val centroids = model.clusterCenters
    val broadcastCentroids = spark.sparkContext.broadcast(centroids)
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
    var totalTime: Double = 0
    // for (i <- 1 to iterations) {
    val startTime = System.nanoTime()
    val futures: Seq[Future[Unit]] = queryFeatures.zipWithIndex.map { case (queryFeature, i) =>
    Future {
      val startTime = System.nanoTime()
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
      val videoClips = filteredRDD.flatMap { case ((videoId, noImg, _), features) =>
      val range = (noImg - 2) to (noImg + 2)
      if (range.exists(i => broadcastedKeys.value.contains((videoId, i)))) {
        Some((videoId, noImg, features))
      } else {
        None
      }
    }
    // 결과 RDD 생성
    val resultRDD: RDD[(String, Long, Seq[(Long, org.apache.spark.ml.linalg.Vector)])] = videoClips.groupBy(_._1).map { case (videoId, clips) =>
      val clipDetails = clips.map{ case (_, noImg, features) => (noImg, features)}.toSeq
      val clipIdStr = clipDetails.map(_._1).mkString(",")
      (clipIdStr, videoId, clipDetails)
    }
    val collectedResults = resultRDD.collect()
    collectedResults.foreach(println)



      // val topKKeys = topKGlobal.map { case ((videoId, noImg), _, _) => (videoId, noImg) }
      // topKKeys.foreach(println)
      // val broadcasttopKKeys = spark.sparkContext.broadcast(topKKeys)
      
      // var resultByTopKKeys = List[((Long, Long, Int), Seq[org.apache.spark.ml.linalg.Vector])]()
  
    
      // broadcasttopKKeys.value.foreach {
      //   case (keyVideoId, keyNoImg) =>
      //     // 현재 topKKey에 해당하는 필터링된 값들의 Vector 추출
      //     val vectorsForKey = filteredRDD
      //       .filter {
      //         case ((currentVideoId, noImg, _), _) =>
      //           keyVideoId == currentVideoId && (keyNoImg >= noImg - 2 && keyNoImg <= noImg + 2)
      //       }
      //       .take(50)  // 원하는 갯수로 수정
      //       .map {
      //         case (_, vector) => vector
      //       }
      //       .toSeq // 추가: 명시적으로 Seq로 변환

      //     // 필터링된 결과를 리스트에 추가
      //     resultByTopKKeys = resultByTopKKeys :+ ((keyVideoId, keyNoImg, 0), vectorsForKey)
      // }

      // // 결과 출력
      // resultByTopKKeys.foreach {
      //   case ((videoId, noImg, idx), vectors) =>
      //     println(s"($videoId, $noImg, $idx): $vectors")
      //     val dtwdistance = dtwDistance(queryFeatures, vectors)
      //     println(s"DTW Distance: $dtwdistance")
      // }
//       var collectedResults = List[((Long, Long, Int), Seq[org.apache.spark.ml.linalg.Vector])]()
//       filteredRDD.foreachPartition { partition =>
//   // 각 파티션에서 필터된 결과를 수집하여 출력하기 위한 리스트
//       var resultByTopKKeysPartition = List[((Long, Long, Int), Seq[org.apache.spark.ml.linalg.Vector])]()

//       // 현재 파티션에서 작업 수행
//       partition.foreach {
//         case ((currentVideoId, noImg, _), vector) =>
//           // 현재 데이터가 topKKeys에 속하는지 확인
//           broadcasttopKKeys.value.foreach {
//             case (keyVideoId, keyNoImg) =>
//               if (keyVideoId == currentVideoId && (keyNoImg >= noImg - 2 && keyNoImg <= noImg + 2)) {
//                 // 필터링된 결과를 리스트에 추가
//                 resultByTopKKeysPartition = resultByTopKKeysPartition :+ ((keyVideoId, keyNoImg, 0), Seq(vector))
//               }
//           }
//       }

//       // 해당 파티션의 결과를 출력
//       resultByTopKKeysPartition.foreach {
//         case ((videoId, noImg, idx), vectors) =>
//           println(s"($videoId, $noImg, $idx): $vectors")
//           val dtwdistance = dtwDistance(queryFeatures, vectors)
//           println(s"DTW Distance: $dtwdistance")
//       }
//     }
//     // val collectedResults: Array[((Long, Long, Int), Seq[org.apache.spark.ml.linalg.Vector])] = filteredRDD.collect()

// // 드라이버에서 수집한 결과를 출력
//     collectedResults.foreach {
//       case ((videoId, noImg, idx), vectors) =>
//         println(s"($videoId, $noImg, $idx): $vectors")
//         val dtwdistance = dtwDistance(queryFeatures, vectors)
//         println(s"DTW Distance: $dtwdistance")
//     }
      // var collectedResults = List[((Long, Long, Int), Seq[org.apache.spark.ml.linalg.Vector])]()
      // val logger: Logger = Logger.getLogger(getClass.getName)
      // filteredRDD.foreachPartition { partition =>
      //   // 각 파티션에서 필터된 결과를 수집하여 출력하기 위한 리스트
      //   // logger.info(s"Results in partition: $resultByTopKKeysPartition")
      //   var resultByTopKKeysPartition = List[((Long, Long, Int), Seq[org.apache.spark.ml.linalg.Vector])]()

      //   // 현재 파티션에서 작업 수행
      //   partition.foreach {
      //     case ((currentVideoId, noImg, _), vector) =>
      //       // 현재 데이터가 topKKeys에 속하는지 확인
      //       broadcasttopKKeys.value.foreach {
      //         case (keyVideoId, keyNoImg) =>
      //           if (keyVideoId == currentVideoId && (keyNoImg >= noImg - 2 && keyNoImg <= noImg + 2)) {
      //             val pw = new PrintWriter("/input/file.txt")
      //             pw.println(s"($currentVideoId, $noImg, 0): $vector")
      //             pw.close()
      //             // 필터링된 결과를 리스트에 추가
      //             // resultByTopKKeysPartition = resultByTopKKeysPartition :+ ((keyVideoId, keyNoImg, 0), Seq(vector))
      //           }
      //       }
      //   }
    //   val collectedResults = filteredRDD.mapPartitions { partition =>
    //   var resultByTopKKeysPartition = List[((Long, Long, Int), Seq[org.apache.spark.ml.linalg.Vector])]()

    //   partition.foreach {
    //     case ((currentVideoId, noImg, _), vector) =>
    //       broadcasttopKKeys.value.foreach {
    //         case (keyVideoId, keyNoImg) =>
    //           if (keyVideoId == currentVideoId && (keyNoImg >= noImg - 2 && keyNoImg <= noImg + 2)) {
    //             resultByTopKKeysPartition = resultByTopKKeysPartition :+ ((keyVideoId, keyNoImg, 0), Seq(vector))
    //           }
    //       }
    //   }

    //   resultByTopKKeysPartition.iterator
    // }.collect()

    // collectedResults.foreach {
    //   case ((videoId, noImg, idx), vectors) =>
    //     println(s"($videoId, $noImg, $idx): $vectors")
    //     val dtwdistance = dtwDistance(queryFeatures, vectors)
    //     println(s"DTW Distance: $dtwdistance")
    // }

        // 현재 파티션의 결과를 저장
        // import spark.implicits._
        // this.synchronized {
        // collectedResults = collectedResults ++ resultByTopKKeysPartition
      // }
      // }

      // 저장된 결과를 출력
      // collectedResults.foreach {
      //   case ((videoId, noImg, idx), vectors) =>
      //     println(s"($videoId, $noImg, $idx): $vectors")
      //     val dtwdistance = dtwDistance(queryFeatures, vectors)
      //     println(s"DTW Distance: $dtwdistance")
      // }
      val endTime = System.nanoTime()
      // val elapsedTime = (endTime - startTime) / 1e9
      // println(s"$i Elapsed time: $elapsedTime seconds")
    }(ExecutionContext.global)
    }
    Await.result(Future.sequence(futures), Duration.Inf)
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    println(s" Elapsed time: $elapsedTime seconds")
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