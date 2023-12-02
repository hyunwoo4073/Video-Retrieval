#!/bin/bash

# Maven 빌드 수행
mvn clean install

# Maven 빌드가 성공적으로 완료되었는지 확인
if [ $? -eq 0 ]; then
  # Spark 애플리케이션을 실행
  spark-submit \
    --deploy-mode client \
    --class com.example.clustering.App \
    --conf spark.kryoserializer.buffer.max.mb=1024 \
    --num-executors 120 \
    ./target/clustering-1.0-SNAPSHOT.jar
else
  echo "Maven 빌드가 실패했습니다. 스크립트를 중지합니다."
fi
