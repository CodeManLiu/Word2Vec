package demo

import java.io.File

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.nlp._
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object THUCNews {
  def main(args: Array[String]): Unit = {
    val trainPath = "C:\\Corpus\\segment\\CNews\\train"
    val testPath = "C:\\Corpus\\segment\\CNews\\test"
    val conf = new SparkConf().setMaster("local[*]").setAppName("THUCNews")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    finalTest(trainPath, testPath, spark)

    spark.stop()
  }

  def finalTest(trainPath: String, testPath: String, spark: SparkSession): Unit ={
    val trainData = spark.read.parquet(trainPath).repartition(10)
    val testData = spark.read.parquet(testPath).repartition(10)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    val word2vec = new Word2Vec().setInputCol("token").setOutputCol("features")
      .setCBOW(0)
      .setHs(true)
      .setNegative(0)
      .setVectorSize(400)
      .setNumPartitions(10)
    val mlpc = new MultilayerPerceptronClassifier().setLayers(Array(400, 50, 10))
    val pipeline = new Pipeline().setStages(Array(tokenizer, word2vec, mlpc))
    val paramGrid = new ParamGridBuilder()
      .addGrid(word2vec.sample, Array(0, 0.1, 0.001, 0.0001))
      .addGrid(word2vec.windowSize, Array(5, 10 , 15, 20, 25))
      .addGrid(word2vec.maxIter, Array(1, 2, 3, 4, 5))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator("accuracy"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
    val model = cv.fit(trainData).bestModel
    model.params.foreach(println)
    val predictionAndLabels = model.transform(testData)
      .select("prediction", "label").rdd
      .map(row => (row.getDouble(0), row.getDouble(1)))
    val metrics = new MulticlassMetrics(predictionAndLabels)
    println(s"acc : ${metrics.accuracy}")
  }

  def word2vec(trainPath: String, testPath: String, spark: SparkSession): Unit = {
    //数据集
    val trainData = spark.read.parquet(trainPath).repartition(10)
    val testData = spark.read.parquet(testPath).repartition(10)
    val modelParam = Array((true,0), (false, 5), (true, 5))
    val acc = Array.ofDim[Double](3,3)
    val time = Array.ofDim[Double](3,3)

    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    trainData.cache()
    testData.cache()
    for(i <- 0 to 2){
      for(j <- 0 to 2){
        time(i)(j) = Double.MaxValue
        acc(i)(j) = 0d
        for(_ <- 0 to 4){
          val start = System.currentTimeMillis()
          val word2vec = new Word2Vec().setInputCol("token").setOutputCol("features")
            .setCBOW(i)
            .setHs(modelParam(j)._1)
            .setSample(0.001)
            .setNegative(modelParam(j)._2)
            .setWindowSize(15)
            .setVectorSize(400)
            .setNumPartitions(10)
            .setMaxIter(1)
          val mlpc = new MultilayerPerceptronClassifier().setLayers(Array(400, 50, 10))
          val pipeline = new Pipeline().setStages(Array(tokenizer, word2vec, mlpc))
          val model = pipeline.fit(trainData)
          val predictionAndLabels = model.transform(testData)
            .select("prediction", "label").rdd
            .map(row => (row.getDouble(0), row.getDouble(1)))
          val metrics = new MulticlassMetrics(predictionAndLabels)
          val now = System.currentTimeMillis()
          val run = (now - start + 1).toDouble / 1000
          acc(i)(j) = math.max(acc(i)(j), metrics.accuracy)
          time(i)(j) = math.min(time(i)(j), run)
        }
      }
    }
    trainData.unpersist()
    testData.unpersist()
    for(i <- 0 to 2){
      for(j <- 0 to 2){
        println(s"$i, $j, acc: ${acc(i)(j)}, time: ${time(i)(j)}")
      }
    }
  }
  def w2v(trainPath: String, testPath: String, spark: SparkSession): Unit = {
    //数据集
    val trainData = spark.read.parquet(trainPath).repartition(10)
    val testData = spark.read.parquet(testPath).repartition(10)
    //训练基础词向量模型
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    import org.apache.spark.ml.feature
    val word2vec = new feature.Word2Vec().setInputCol("token").setOutputCol("features")
      .setWindowSize(15)
      .setVectorSize(400)
      .setNumPartitions(10)
      .setMaxIter(1)
    val mlpc = new MultilayerPerceptronClassifier().setLayers(Array(400, 50, 10))
    val pipeline = new Pipeline().setStages(Array(tokenizer, word2vec, mlpc))
    trainData.cache()
    testData.cache()
    var acc = 0d
    var time = Double.MaxValue
    for(_ <- 0 to 4) {
      val start = System.currentTimeMillis()
      val model = pipeline.fit(trainData)
      val predictionAndLabels = model.transform(testData)
        .select("prediction", "label").rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val now = System.currentTimeMillis()
      val run = (now - start + 1).toDouble / 1000
      acc = math.max(metrics.accuracy, acc)
      time = math.min(time, run)
    }
    trainData.unpersist()
    testData.unpersist()
    println(s"acc : $acc, time: $time")
  }
  def tf_idf(trainPath: String, testPath: String,spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath).repartition(8)
    val testData = spark.read.parquet(testPath).repartition(8)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    val tf = new HashingTF().setInputCol("token").setOutputCol("tf")
      .setNumFeatures(10000)
    val idf = new IDF().setInputCol("tf").setOutputCol("features")
    val mlpc = new MultilayerPerceptronClassifier().setLayers(Array(10000, 50, 10))
    val pipeline = new Pipeline().setStages(Array(tokenizer, tf, idf, mlpc))

    trainData.cache()
    testData.cache()
    var acc = 0d
    var time = Double.MaxValue
    for(_ <- 0 to 4) {
      val start = System.currentTimeMillis()
      val model = pipeline.fit(trainData)
      val predictionAndLabels = model.transform(testData)
        .select("prediction", "label").rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val now = System.currentTimeMillis()
      val run = (now - start + 1).toDouble / 1000
      acc = math.max(metrics.accuracy, acc)
      time = math.min(time, run)
    }
    trainData.unpersist()
    testData.unpersist()
    println(s"acc : $acc, time: $time")
  }
  def doc2vec(trainPath: String, testPath: String, spark: SparkSession): Unit = {
    import spark.implicits._
    val trainData = spark.read.parquet(trainPath).repartition(10)
      .rdd.zipWithIndex()
      .map { case (row, id) => (id.toInt, row.getDouble(0), row.getString(1)) }
      .toDF("id", "label", "doc")
    val testData = spark.read.parquet(testPath).repartition(10)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    val d2v = new Doc2Vec().setInputCol("token").setOutputCol("features").setDocIdCol("id")
      .setStepSize(0.025)
      .setWindow(15)
      .setDecay(0.85)
      .setSample(0.01)
      .setVectorSize(400)
      .setMaxIter(1)
      .setNumPartitions(10)
    val mlpc = new MultilayerPerceptronClassifier().setLayers(Array(400, 50, 10))
    val pipeline = new Pipeline().setStages(Array(tokenizer, d2v, mlpc))
    trainData.cache()
    testData.cache()
    var acc =0d
    var time = Double.MaxValue
    for(_ <- 0 to 4) {
      val start = System.currentTimeMillis()
      val model = pipeline.fit(trainData)
      val predictionAndLabels = model.transform(testData)
        .select("prediction", "label").rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val now = System.currentTimeMillis()
      val run = (now - start + 1).toDouble / 1000
      acc = math.max(metrics.accuracy, acc)
      time = math.min(time, run)
    }
    trainData.unpersist()
    testData.unpersist()
    println(s"acc : $acc, time: $time")
  }

  //将原始THUNews分类新闻提取label并统一到一个文件中。
  def merge(inputPath: String, outputPath: String, spark: SparkSession): Unit = {
    val sc = spark.sparkContext
    import spark.implicits._

    val dir = new File(inputPath).listFiles()
    val totalDocDF = {
      for {
        file <- dir
        docWithPathRDD = sc.wholeTextFiles(file.toString + "\\*", 12)
      } yield docWithPathRDD
    }
      .reduce(_ ++ _)
      .map { case (path, doc) =>
        println(path)
        (path.split("/")(4).toDouble, doc)
      }
      .repartition(12)
      .toDF("label", "doc")
    totalDocDF.write.parquet(outputPath)
  }
  //对THUNews类统一的含有label和doc的文档进行Hanlp分词
  def segment(inputPath: String, outputPath: String, stopWordsPath: String, spark: SparkSession): Unit = {
    val sc = spark.sparkContext
    import spark.implicits._
    val bcStopWordSet = sc.broadcast(sc.textFile(stopWordsPath).collect().toSet)
    val segRDD = spark.read.parquet(inputPath).rdd
      .map(row => (row.getDouble(0), row.getString(1)))
      .mapPartitions { iter =>
        val segment = new NShortSegment()
          .enableCustomDictionary(true)
          .enableAllNamedEntityRecognize(true)
        iter.map { case (label, doc) =>
          val segRS = doc.split("\\n")
            .map { line =>
              segment.seg(line).asScala
                .map(_.word.replaceAll(" +", ""))
                .filter(!_.isEmpty)
                .filter(!bcStopWordSet.value.contains(_))
                .mkString(" ")
            }
            .mkString(" ")
          (label, segRS)
        }
      }
    segRDD.toDF("label", "doc").write.parquet(outputPath)
    bcStopWordSet.destroy()
  }
  def segCNews(inputPath: String, outputPath: String, stopWordsPath: String, spark: SparkSession): Unit = {
    val sc = spark.sparkContext
    val labelMap = Map(("体育", 0d), ("财经", 1d), ("房产", 2d), ("家居", 3d), ("教育", 4d), ("科技", 5d),
      ("时尚", 6d), ("时政", 7d), ("游戏", 8d), ("娱乐", 9d))
    val bcStopWordSet = sc.broadcast(sc.textFile(stopWordsPath).collect().toSet)
    val segRDD = sc.textFile(inputPath, 10)
      .map(line => (line.slice(0, 2), line.slice(3, line.length)))
      .map(line => (labelMap(line._1), line._2))
      .mapPartitions { iter =>
        val segment = new NShortSegment()
          .enableAllNamedEntityRecognize(true)
          .enableCustomDictionary(true)
        iter.map { case (label, doc) =>
          val segRS = doc.split("\\n")
            .map { line =>
              segment.seg(line).asScala
                .map(_.word.replaceAll(" +", ""))
                .filter(!_.isEmpty)
                .filter(!bcStopWordSet.value.contains(_))
                .mkString(" ")
            }
            .mkString(" ")
          (label, segRS)
        }
      }
    import spark.implicits._
    segRDD.toDF("label", "doc").write.parquet(outputPath)
    bcStopWordSet.destroy()
  }
  def splitData(inputPath: String, trainPath: String, testPath: String, spark: SparkSession): Unit ={
    val allData = spark.read.parquet(inputPath)
    val splitData = for { i <- 0 to 13
      classData = allData.filter(s"label = $i").randomSplit(Array(0.8, 0.2))
    } yield classData
    val finalData = splitData.reduce((a, b) =>
     Array(a(0).union(b(0)), b(1).union(b(1))))
    val trainData = finalData(0)
    val testData = finalData(1)
    trainData.write.parquet(trainPath)
    testData.write.parquet(testPath)
  }
}