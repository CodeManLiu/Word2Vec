package demo

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.nlp._
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object IMDB {
  def main(args: Array[String]): Unit = {
    val trainPath = "D:\\毕设相关\\corpus\\IMDB\\token\\train"
    val testPath = "D:\\毕设相关\\corpus\\IMDB\\token\\test"
    val conf = new SparkConf().setAppName("IMDB").setMaster("local[*]")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    word2vec(trainPath, testPath, spark)
    spark.stop()
  }

  def testW2V(trainPath: String, testPath: String, spark: SparkSession): Unit ={
    val trainData = spark.read.parquet(trainPath).repartition(10)
    val trainDF = trainData.filter("label != 2")
    val testDF = spark.read.parquet(testPath).repartition(10)
    val param = Array(1, 2, 3, 4, 5, 6)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    trainData.cache()
    testDF.cache()
    for (i <- 0 to 5) {
      val start = System.currentTimeMillis()
      import org.apache.spark.ml.feature
      val word2vec = new feature.Word2Vec().setInputCol("token").setOutputCol("features")
        .setVectorSize(400)
        .setWindowSize(15)
        .setNumPartitions(10)
        .setMaxIter(param(i))
      val w2vModel = word2vec.fit(tokenizer.transform(trainData))
      val lr = new LogisticRegression().setRegParam(0.01)
      val pipeline = new Pipeline().setStages(Array(tokenizer, w2vModel, lr))
      val model = pipeline.fit(trainDF)
      val predictionAndLabels = model.transform(testDF)
        .select("prediction", "label").rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val now = System.currentTimeMillis()
      val time = (now - start + 1).toDouble / 1000
      val acc = metrics.accuracy
      println(s"$i, acc: $acc, time:$time")
    }
    trainData.unpersist()
    testDF.unpersist()
  }
  def d2vTestVectorSize(trainPath: String, testPath: String, spark: SparkSession): Unit ={
    import spark.implicits._
    val trainData = spark.read.parquet(trainPath).repartition(10)
      .rdd.zipWithIndex()
      .map { case (row, id) => (id.toInt, row.getDouble(0), row.getString(2)) }
      .toDF("id", "label", "doc")
    val trainDF = trainData.filter("label != 2").select("label", "doc")
    val testDF = spark.read.parquet(testPath).repartition(10)
    val param = Array(100, 200, 300, 400, 500, 600)
    val acc = new Array[Double](6)
    val time = new Array[Double](6)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    trainData.cache()
    testDF.cache()
    for(i <- 0 to 5){
      val start = System.currentTimeMillis()
      val doc2vec = new Doc2Vec().setInputCol("token").setOutputCol("features")
        .setWindow(15)
        .setVectorSize(param(i))
        .setSample(0.001)
        .setStepSize(0.025)
        .setMinCount(5)
        .setNumPartitions(10)
        .setMaxIter(1)
      val d2vModel = doc2vec.fit(tokenizer.transform(trainData))
      val lr = new LogisticRegression().setRegParam(0.01)
      val pipeline = new Pipeline().setStages(Array(tokenizer, d2vModel, lr))
      val model = pipeline.fit(trainDF)
      val predictionAndLabels = model.transform(testDF)
        .select("prediction", "label").rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val now = System.currentTimeMillis()
      val run = (now - start + 1).toDouble / 1000
      acc(i) = metrics.accuracy
      time(i) = run
    }
    trainData.unpersist()
    testDF.unpersist()
    for(i <- 0 to 5){
      println(s"$i, acc: ${acc(i)}, time:${time(i)}")
    }
  }
  def d2vTestWindow(trainPath: String, testPath: String, spark: SparkSession): Unit ={
    import spark.implicits._
    val trainData = spark.read.parquet(trainPath).repartition(10)
      .rdd.zipWithIndex()
      .map { case (row, id) => (id.toInt, row.getDouble(0), row.getString(2)) }
      .toDF("id", "label", "doc")
    val trainDF = trainData.filter("label != 2").select("label", "doc")
    val testDF = spark.read.parquet(testPath).repartition(10)
    val param = Array(5, 10, 15, 20, 25, 30)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    trainData.cache()
    testDF.cache()
    for(i <- 0 to 5){
      val start = System.currentTimeMillis()
      val doc2vec = new Doc2Vec().setInputCol("token").setOutputCol("features")
        .setWindow(param(i))
        .setVectorSize(400)
        .setSample(0.001)
        .setStepSize(0.025)
        .setMinCount(5)
        .setNumPartitions(10)
        .setMaxIter(1)
      val d2vModel = doc2vec.fit(tokenizer.transform(trainData))
      val lr = new LogisticRegression().setRegParam(0.01)
      val pipeline = new Pipeline().setStages(Array(tokenizer, d2vModel, lr))
      val model = pipeline.fit(trainDF)
      val predictionAndLabels = model.transform(testDF)
        .select("prediction", "label").rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val now = System.currentTimeMillis()
      val time = (now - start + 1).toDouble / 1000
      val acc = metrics.accuracy
      println(s"$i, acc: $acc, time:$time")
    }
    trainData.unpersist()
    testDF.unpersist()
  }
  def d2vTestInter(trainPath: String, testPath: String, spark: SparkSession): Unit ={
    import spark.implicits._
    val trainData = spark.read.parquet(trainPath).repartition(10)
      .rdd.zipWithIndex()
      .map { case (row, id) => (id.toInt, row.getDouble(0), row.getString(2)) }
      .toDF("id", "label", "doc")
    val trainDF = trainData.filter("label != 2").select("label", "doc")
    val testDF = spark.read.parquet(testPath).repartition(10)
    val param = Array(1, 2, 3, 4, 5, 6)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    trainData.cache()
    testDF.cache()
    for(i <- 0 to 5){
      val start = System.currentTimeMillis()
      val doc2vec = new Doc2Vec().setInputCol("token").setOutputCol("features")
        .setWindow(15)
        .setVectorSize(400)
        .setSample(0.001)
        .setStepSize(0.025)
        .setMinCount(5)
        .setNumPartitions(10)
        .setMaxIter(param(i))
      val d2vModel = doc2vec.fit(tokenizer.transform(trainData))
      val lr = new LogisticRegression().setRegParam(0.01)
      val pipeline = new Pipeline().setStages(Array(tokenizer, d2vModel, lr))
      val model = pipeline.fit(trainDF)
      val predictionAndLabels = model.transform(testDF)
        .select("prediction", "label").rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val now = System.currentTimeMillis()
      val time = (now - start + 1).toDouble / 1000
      val acc = metrics.accuracy
      println(s"$i, acc: $acc, time:$time")
    }
    trainData.unpersist()
    testDF.unpersist()
  }
  def testInter(trainPath: String, testPath: String, spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath).repartition(10)
    val trainDF = trainData.filter("label != 2")
    val testDF = spark.read.parquet(testPath).repartition(10)
    val interParam = Array(1, 2, 3, 4, 5, 6)
    val acc = new Array[Double](6)
    val time = new Array[Double](6)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")

    trainData.cache()
    testDF.cache()
    for (i <- 0 to 5) {
      time(i) = Double.MaxValue
      acc(i) = 0d
      for (_ <- 0 to 4) {
        val start = System.currentTimeMillis()
        val word2vec = new Word2Vec().setInputCol("token").setOutputCol("features")
          .setCBOW(0)
          .setHs(true)
          .setNegative(0)
          .setSample(0.001)
          .setVectorSize(400)
          .setWindowSize(15)
          .setNumPartitions(10)
          .setMaxIter(interParam(i))
        val w2vModel = word2vec.fit(tokenizer.transform(trainData))
        val lr = new LogisticRegression().setRegParam(0.01)
        val pipeline = new Pipeline().setStages(Array(tokenizer, w2vModel, lr))
        val model = pipeline.fit(trainDF)
        val predictionAndLabels = model.transform(testDF)
          .select("prediction", "label").rdd
          .map(row => (row.getDouble(0), row.getDouble(1)))
        val metrics = new MulticlassMetrics(predictionAndLabels)
        val now = System.currentTimeMillis()
        val run = (now - start + 1).toDouble / 1000
        acc(i) = math.max(metrics.accuracy, acc(i))
        time(i) = math.min(run, time(i))
      }
    }
    trainData.unpersist()
    testDF.unpersist()
    for(i <- 0 to 5){
      println(s"$i, acc: ${acc(i)}, time:${time(i)}")
    }
  }
  def testSample(trainPath: String, testPath: String, spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath).repartition(10)
    val trainDF = trainData.filter("label != 2")
    val testDF = spark.read.parquet(testPath).repartition(10)
    val sampleParam = Array(0.00001, 0.0001, 0.001, 0.01, 0.1, 0)
    val acc = new Array[Double](6)
    val time = new Array[Double](6)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")

    trainData.cache()
    testDF.cache()
    for (i <- 0 to 5) {
      time(i) = Double.MaxValue
      acc(i) = 0d
      for (_ <- 0 to 4) {
        val start = System.currentTimeMillis()
        val word2vec = new Word2Vec().setInputCol("token").setOutputCol("features")
          .setCBOW(0)
          .setHs(true)
          .setNegative(0)
          .setSample(sampleParam(i))
          .setVectorSize(400)
          .setWindowSize(15)
          .setNumPartitions(10)
          .setMaxIter(1)
        val w2vModel = word2vec.fit(tokenizer.transform(trainData))
        val lr = new LogisticRegression().setRegParam(0.01)
        val pipeline = new Pipeline().setStages(Array(tokenizer, w2vModel, lr))
        val model = pipeline.fit(trainDF)
        val predictionAndLabels = model.transform(testDF)
          .select("prediction", "label").rdd
          .map(row => (row.getDouble(0), row.getDouble(1)))
        val metrics = new MulticlassMetrics(predictionAndLabels)
        val now = System.currentTimeMillis()
        val run = (now - start + 1).toDouble / 1000
        acc(i) = math.max(metrics.accuracy, acc(i))
        time(i) = math.min(run, time(i))
      }
    }
    trainData.unpersist()
    testDF.unpersist()
    for(i <- 0 to 5){
      println(s"$i, acc: ${acc(i)}, time:${time(i)}")
    }
  }
  def testVectorSize(trainPath: String, testPath: String, spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath).repartition(10)
    val trainDF = trainData.filter("label != 2")
    val testDF = spark.read.parquet(testPath).repartition(10)
    val vectorSizeParam = Array(100, 200, 300, 400, 500, 600)
    val acc = new Array[Double](6)
    val time = new Array[Double](6)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")

    trainData.cache()
    testDF.cache()
    for (i <- 0 to 5) {
      time(i) = Double.MaxValue
      acc(i) = 0d
      for (_ <- 0 to 4) {
        val start = System.currentTimeMillis()
        val word2vec = new Word2Vec().setInputCol("token").setOutputCol("features")
          .setCBOW(0)
          .setHs(true)
          .setNegative(0)
          .setSample(0.001)
          .setVectorSize(vectorSizeParam(i))
          .setWindowSize(15)
          .setNumPartitions(10)
          .setMaxIter(1)
        val w2vModel = word2vec.fit(tokenizer.transform(trainData))
        val lr = new LogisticRegression().setRegParam(0.01)
        val pipeline = new Pipeline().setStages(Array(tokenizer, w2vModel, lr))
        val model = pipeline.fit(trainDF)
        val predictionAndLabels = model.transform(testDF)
          .select("prediction", "label").rdd
          .map(row => (row.getDouble(0), row.getDouble(1)))
        val metrics = new MulticlassMetrics(predictionAndLabels)
        val now = System.currentTimeMillis()
        val run = (now - start + 1).toDouble / 1000
        acc(i) = math.max(metrics.accuracy, acc(i))
        time(i) = math.min(run, time(i))
      }
    }
    trainData.unpersist()
    testDF.unpersist()
    for(i <- 0 to 5){
      println(s"$i, acc: ${acc(i)}, time:${time(i)}")
    }
  }
  def testWindow(trainPath: String, testPath: String, spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath).repartition(10)
    val trainDF = trainData.filter("label != 2")
    val testDF = spark.read.parquet(testPath).repartition(10)
    val windowParam = Array(5, 10, 15, 20, 25, 30)
    val acc = new Array[Double](6)
    val time = new Array[Double](6)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")

    trainData.cache()
    testDF.cache()
    for (i <- 0 to 5) {
      time(i) = Double.MaxValue
      acc(i) = 0d
      for (_ <- 0 to 4) {
        val start = System.currentTimeMillis()
        val word2vec = new Word2Vec().setInputCol("token").setOutputCol("features")
          .setCBOW(0)
          .setHs(true)
          .setNegative(0)
          .setSample(0.001)
          .setVectorSize(400)
          .setWindowSize(windowParam(i))
          .setNumPartitions(10)
          .setMaxIter(1)
        val w2vModel = word2vec.fit(tokenizer.transform(trainData))
        val lr = new LogisticRegression().setRegParam(0.01)
        val pipeline = new Pipeline().setStages(Array(tokenizer, w2vModel, lr))
        val model = pipeline.fit(trainDF)
        val predictionAndLabels = model.transform(testDF)
          .select("prediction", "label").rdd
          .map(row => (row.getDouble(0), row.getDouble(1)))
        val metrics = new MulticlassMetrics(predictionAndLabels)
        val now = System.currentTimeMillis()
        val run = (now - start + 1).toDouble / 1000
        acc(i) = math.max(metrics.accuracy, acc(i))
        time(i) = math.min(run, time(i))
      }
    }
    trainData.unpersist()
    testDF.unpersist()
    for(i <- 0 to 5){
      println(s"$i, acc: ${acc(i)}, time:${time(i)}")
    }
  }

  def word2vec(trainPath: String, testPath: String, spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath).repartition(10)
    val trainDF = trainData.filter("label != 2")
    val testDF = spark.read.parquet(testPath).repartition(10)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")

    trainData.cache()
    val start = System.currentTimeMillis()
    val word2vec = new Word2Vec().setInputCol("token").setOutputCol("features")
      .setCBOW(0)
      .setHs(true)
      .setNegative(0)
      .setSample(0.001)
      .setVectorSize(400)
      .setWindowSize(15)
      .setNumPartitions(1)
      .setMaxIter(1)
    val w2vModel = word2vec.fit(tokenizer.transform(trainData))
    val lr = new LogisticRegression().setRegParam(0.01)
    val pipeline = new Pipeline().setStages(Array(tokenizer, w2vModel, lr))
    val model = pipeline.fit(trainDF)
    val predictionAndLabels = model.transform(testDF)
      .select("prediction", "label").rdd
      .map(row => (row.getDouble(0), row.getDouble(1)))
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val now = System.currentTimeMillis()
    val run = (now - start + 1).toDouble / 1000
    println(s"acc : ${metrics.accuracy}, time : $run")
    trainData.unpersist()
  }

  def w2v(trainPath: String, testPath: String, spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath).repartition(10)
    val trainDF = trainData.filter("label != 2")
    val testDF = spark.read.parquet(testPath).repartition(10)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    import org.apache.spark.ml.feature
    val word2vec = new feature.Word2Vec().setInputCol("token").setOutputCol("features")
      .setVectorSize(400)
      .setWindowSize(15)
      .setNumPartitions(10)
      .setMaxIter(1)
    var acc = 0d
    var time = Double.MaxValue
    trainData.cache()
    testDF.cache()
    for(_ <- 0 to 4) {
      val start = System.currentTimeMillis()
      val w2vModel = word2vec.fit(tokenizer.transform(trainData))
      val lr = new LogisticRegression().setRegParam(0.01)
      val pipeline = new Pipeline().setStages(Array(tokenizer, w2vModel, lr))
      val model = pipeline.fit(trainDF)

      val predictionAndLabels = model.transform(testDF)
        .select("prediction", "label").rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val now = System.currentTimeMillis()
      val run = (now - start + 1).toDouble / 1000
      acc = math.max(metrics.accuracy, acc)
      time = math.min(run, time)
    }
    trainData.unpersist()
    testDF.unpersist()
    println(s"acc : $acc, time: $time")
  }
  def tf_idf(trainPath: String, testPath: String,spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath).repartition(10)
    val trainDF = trainData.filter("label != 2")
    val testDF = spark.read.parquet(testPath).repartition(10)
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    val tf = new HashingTF().setInputCol("token").setOutputCol("tf")
      .setNumFeatures(10000)
    val idf = new IDF().setInputCol("tf").setOutputCol("features")
    var acc= 0d
    var time = Double.MaxValue
    trainData.cache()
    testDF.cache()
    for(_ <- 0 to 4){
      val start = System.currentTimeMillis()
      val idfModel = idf.fit(tf.transform(tokenizer.transform(trainData)))
      val lr = new LogisticRegression().setRegParam(0.01)
      val pipeline = new Pipeline().setStages(Array(tokenizer, tf, idfModel, lr))
      val model = pipeline.fit(trainDF)
      val predictionAndLabels = model.transform(testDF)
        .select("prediction", "label").rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val now = System.currentTimeMillis()
      val run = (now - start + 1).toDouble / 1000
      acc = math.max(metrics.accuracy, acc)
      time = math.min(time, run)
    }
    trainData.unpersist()
    testDF.unpersist()
    println(s"acc : $acc, time: $time")
  }
  def doc2vec(trainPath: String, testPath: String, spark: SparkSession): Unit = {
    import spark.implicits._
    val trainData = spark.read.parquet(trainPath).repartition(10)
      .rdd.zipWithIndex()
      .map { case (row, id) => (id.toInt, row.getDouble(0), row.getString(2)) }
      .toDF("id", "label", "doc")
    val trainDF = trainData.filter("label != 2").select("label", "doc")
    val testDF = spark.read.parquet(testPath).repartition(10)

    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    val doc2vec = new Doc2Vec().setInputCol("token").setOutputCol("features").setDocIdCol("id")
      .setStepSize(0.025)
      .setSample(0.001)
      .setWindow(15)
      .setDecay(0.75)
      .setVectorSize(400)
      .setNumPartitions(10)
      .setMaxIter(1)
    var acc = 0d
    var time = Double.MaxValue
    trainData.cache()
    testDF.cache()
    for(_ <- 0 to 4) {
      val start = System.currentTimeMillis()
      val d2vModel = doc2vec.fit(tokenizer.transform(trainData))
      val lr = new LogisticRegression().setRegParam(0.01)
      val pipeline = new Pipeline().setStages(Array(tokenizer, d2vModel, lr))
      val model = pipeline.fit(trainDF.filter("label != 2"))
      val predictionAndLabels = model.transform(testDF)
        .select("prediction", "label").rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val now = System.currentTimeMillis()
      val run = (now - start + 1).toDouble / 1000
      acc = math.max(metrics.accuracy, acc)
      time = math.min(time, run)
    }
    trainData.unpersist()
    testDF.unpersist()
    println(s"acc : $acc, time: $time")
  }

  def segment(inputPath: String, outputPath: String, label: Int, ty: String, spark: SparkSession): Unit ={
    val sc = spark.sparkContext
    import spark.implicits._
    val fileRDD = sc.wholeTextFiles(inputPath, 12)
    val segRDD = fileRDD.map { case (fileName, content) =>
      val doc = new Document(content)
      val text = doc.sentences()
        .asScala.map { sent =>
        sent.words().asScala
          .filter(!_.matches("<br.*/>"))
          .mkString(" ")
        }.mkString(" ")
      val split = fileName.split("/")(6).replace(".txt", "").split("_")
      (split(0).toInt, ty, label.toDouble, split(1).toDouble, text)
    }
    segRDD.toDF("id", "type", "label", "rate", "doc").write.parquet(outputPath)
  }
  def merge(inputPath: Array[String], outputPath: String, spark: SparkSession): Unit = {
    import spark.implicits._

    val mergeRDD = for {
      path <- inputPath
      fileRDD = spark.read.parquet(path).rdd
        .map(row => (row.getInt(0), row.getString(1), row.getDouble(2), row.getDouble(3), row.getString(4)))
    } yield fileRDD
    mergeRDD.reduce(_ ++ _)
      .zipWithIndex()
      .mapValues(_.toInt)
      .map { case ((_, ty, label, rate, doc), id) =>
        (id, ty, label, rate, doc)
      }
      .toDF("id", "type", "label", "rate", "doc")
      .write.parquet(outputPath)
  }
}
