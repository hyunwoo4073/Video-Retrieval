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
import scala.collection.mutable.ArrayBuffer
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
    val numClusters = 1000
    val kmeans = new KMeans().setK(numClusters).setSeed(1L)
    // val model = kmeans.fit(featureData)
    val model = KMeansModel.load("/input/km_model/")
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
    val iterations = 10
    // var totalTime: Double = 0
    case class ResultRow(videoId: Long, noImg: Long, origFeatures: Vector)
    // for (i <- 1 to iterations) {

    val startTime = System.nanoTime()
    // get 2 closest centroids to the query features
    val closestClusterIds = broadcastCentroids.value.zipWithIndex.map { case (center, idx) =>
      (idx, Vectors.sqdist(center, queryFeatures))
    }.sortBy(_._2).take(10).map(_._1)
    // closestClusterIds.foreach(println)
    val filteredRDD = partitionedRDD.filter { case ((_, _, clusterId), _) =>
      closestClusterIds.contains(clusterId)
    }.partitionBy(new ClusterPartitioner(numClusters))
    .cache()
    val topKPartitions = filteredRDD
      .mapPartitions(iter => topKIterator(iter, k), preservesPartitioning = true)
    val topKGlobal = topKPartitions
      .reduce((a, b) => (a ++ b).sortBy(-_._2).take(k))

    // topKGlobal.foreach(println)

    val topKKeys = topKGlobal.map { case ((videoId, noImg), _, _) => (videoId, noImg) }
    val broadcasttopKKeys = spark.sparkContext.broadcast(topKKeys)
    broadcasttopKKeys.value.foreach(println)
    // topKKeys.foreach(println)
    // val videoclip = filteredRDD.filter { case ((videoId, noImg, _), _) => 
    //   topKKeys.contains(videoId, noImg)
    // }
    // val videoclip = filteredRDD.filter {
    //   case ((videoId, noImg, _), _) =>
    //     topKKeys.exists { case (keyVideoId, keyNoImg) =>
    //       keyVideoId == videoId && (keyNoImg >= noImg - 2 && keyNoImg <= noImg + 2)
    //     }
    // }
    // val sortedVideoclip = videoclip.sortBy {
    //   case ((videoId, noImg, _), _) => (videoId, noImg)
    // }
    // val videoclipnum = sortedVideoclip.count()
    // print(s"videoclip: $videoclipnum ")
    var resultByTopKKeys = List[((Long, Long, Int), Vector)]()

// topKKeys를 순회하면서 필터링하고 결과를 리스트에 추가
    broadcasttopKKeys.value.foreach {
      case (keyVideoId, keyNoImg) =>
        val filteredVideoclip = filteredRDD.filter {
          case ((currentVideoId, noImg, _), _) =>
            keyVideoId == currentVideoId && (keyNoImg >= noImg - 2 && keyNoImg <= noImg + 2)
        }

        // 필터링된 결과를 리스트에 추가
        resultByTopKKeys = resultByTopKKeys ::: filteredVideoclip.map {
          case ((currentVideoId, noImg, idx), vec) => ((currentVideoId, noImg, idx), vec)
        }.collect().toList
    }

    // 결과 출력
    resultByTopKKeys.foreach(println)
    
    println(s"Number of elements in resultByTopKKeys: ${resultByTopKKeys.size}")
    // 결과 출력
    // println(topKKeys.take(1).getClass.getName)





    // foreachPartition을 이용하여 작업 수행
//     topKKeys.foreachPartition { iter =>
//       iter.foreach { case (videoId, noImg) =>
//         // 해당 videoId, noImg에 대한 조건을 만족하는 row들을 찾음
//         val matchingRows = filteredRDD.filter { case ((vId, nImg, _), _) =>
//           vId == videoId && nImg == noImg
//         }

//         // 찾은 row들을 정렬하여 앞으로 2개, 뒤로 2개를 선택하거나, 조건에 따라 처리
//         val window: Option[Vector[((Long, Long, Int), Vector)]] = matchingRows.collect().sortBy { case ((_, _, index), _) => index } match {
//           case Array(first, second, _, _, _) => Some(Array(first, second, matchingRows(2), matchingRows(3), matchingRows(4)).toVector)
//           case Array(_, second, third, _, _) => Some(Array(second, third, matchingRows(3), matchingRows(4), matchingRows(5)).toVector)
//           case Array(_, _, third, fourth, _) => Some(Array(third, fourth, matchingRows(4), matchingRows(5), matchingRows(6)).toVector)
//           case Array(_, _, _, fourth, fifth) => Some(Array(fourth, fifth, matchingRows(5), matchingRows(6), matchingRows(7)).toVector)
//           case Array(_, _, _, _, fifth) => Some(Array(fifth, matchingRows(6), matchingRows(7), matchingRows(8), matchingRows(9)).toVector)
//           case _ => None // 앞으로 2개, 뒤로 2개가 없는 경우
//         }

//         // 결과 출력
//         window.foreach { w =>
//           w.foreach { case ((vId, nImg, _), _) =>
//             println(s"videoId: $videoId, noImg: $noImg, vId: $vId, nImg: $nImg")
//           }
//         }
//       }
//     }



// processRDD(topKKeys, filteredRDD)















    // val videoclip = filteredRDD.filter { case ((videoId, noImg, _), _) =>
    //   topKKeys.contains(videoId, noImg)
    // }
  //   val videoclips = topKKeys.flatMap { case (videoId, noImg) =>

  // // filteredRDD에서 해당 (videoId, noImg)를 찾아서 앞뒤 2개씩을 선택
  //   val clips = filteredRDD
  //     .filter { case ((vId, nImg, _), _) => vId == videoId && nImg == noImg }
  //     .mapPartitions { iter =>
  //       // Iterator에서 앞뒤 2개씩 선택
  //       iter.sliding(5).flatMap { window =>
  //         if (window.length == 5) Some(window) else None
  //         window
  //       }

  //     }
  //   }





    // Collect and print the rows of videoclips
    // videoclips.foreach { clip =>
    //   val ((videoId, noImg, clusterId), origFeatures) = clip.head
    //   println(s"($videoId, $noImg, $clusterId): $origFeatures")
    // }

    // videoclip.take(k).foreach(println)
    // topKKeys.foreach(println)
    // Use foreachPartition to perform the desired operation on each partition
  //   partitionedRDD.foreachPartition { iter =>
  // // Filter rows that match topKKeys
  //   val filteredRows = iter.filter { case ((videoId, noImg, _), _) => topKKeys.contains((videoId, noImg)) }

  //   // Collect filteredRows into a list
  //   val rowsList = filteredRows.toList

  //   // Iterate over rowsList and print the desired output
  //   rowsList.sliding(5).foreach { window =>
  //     // window contains 5 consecutive rows
  //     // Extract information from each row and print
  //     window.foreach { case ((videoId, noImg, _), origFeatures) =>
  //       val resultRow = ResultRow(videoId, noImg, origFeatures)
  //       println(resultRow)
  //     }
  //   }
  // }
    // topKKeys.foreach(println)
    // val broadcastTopKKeys = spark.sparkContext.broadcast(topKKeys.toSet)




    // val collectedRows = partitionedRDD.mapPartitions { iter =>
    //   val localCollectedRows = new ArrayBuffer[String]()

    //   // Filter rows in the partition based on topKKeys
    //   val matchingRows = iter.filter { case ((videoId, noImg, _), _) =>
    //     broadcastTopKKeys.value.contains((videoId, noImg))
    //   }

    //   matchingRows.foreach { case ((videoId, noImg, clusterId), origFeatures) =>
    //     localCollectedRows += s"($videoId, $noImg, $clusterId): $origFeatures"
    //   }

    //   Iterator(localCollectedRows)
    // }

    // // // Collect and print the collectedRows from all partitions
    // // collectedRows.collect().foreach(_.foreach(println))

    // // Collect and print the matching rows
    // val keyToRowsMap: Map[(Long, Long), List[((Long, Long, Int), Vector)]] = collectedRows
    //   .groupBy { case ((videoId, noImg, clusterId), _) => (videoId, noImg) }
    //   .mapValues(_.map { case ((videoId, noImg, clusterId), v) => ((videoId, noImg, clusterId), v) }.toList)
    //   .collect()
    //   .toMap

    // // Iterate over topKKeys and print the required rows
    // topKKeys.foreach { case (videoId, noImg) =>
    //   val key = (videoId, noImg)
    //   val rows = keyToRowsMap.getOrElse(key, List.empty)

    //   // If there are rows for the key, print the required 5 rows
    //   if (rows.nonEmpty) {
    //     val indexOfKey = rows.indexWhere { case ((vid, img, _), _) => vid == videoId && img == noImg }
    //     val startIndex = math.max(0, indexOfKey - 2)
    //     val endIndex = math.min(rows.length - 1, indexOfKey + 2)

    //     val selectedRows = rows.slice(startIndex, endIndex + 1)
    //     selectedRows.foreach { case ((vid, img, clusterId), origFeatures) =>
    //       println(s"($vid, $img, $clusterId): $origFeatures")
    //     }
    //   }
    // }
      
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    // println(s"$i Elapsed time: $elapsedTime seconds")
    // totalTime += elapsedTime.toDouble
    
  // }
  
  // val averageTime = totalTime / iterations
    println(s"average time: $elapsedTime seconds")
    spark.stop()
    }
  
}