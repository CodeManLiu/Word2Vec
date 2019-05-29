import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.nlp.Word2Vec
import org.apache.spark.sql.SparkSession

object runModelTrainer {
  def main(args: Array[String]): Unit = {
    val trainPath = args(0)
    val executors = args(1).toInt
    val conf = new SparkConf().setAppName("runModelTrainer").setMaster("local[*]")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    w2v(trainPath, executors, spark)
    SkipHs(trainPath, executors, spark)
    SkipNeg(trainPath, executors, spark)
    CBOWHs(trainPath, executors, spark)
    CBOWNeg(trainPath, executors, spark)
    spark.stop()
  }
  def w2v(trainPath: String, executors: Int, spark: SparkSession): Unit ={
    import org.apache.spark.ml.feature
    val trainData = spark.read.parquet(trainPath)
    val start = System.currentTimeMillis()
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    val model = new feature.Word2Vec().setInputCol("token").setOutputCol("features")
      .setVectorSize(200)
      .setWindowSize(10)
      .setNumPartitions(executors)
      .setMaxIter(1)
      .fit(tokenizer.transform(trainData))
    val now = System.currentTimeMillis()
    val run = (now - start + 1).toDouble / 1000
    println(f"$run%.2f")
  }
  def SkipHs(trainPath: String, executors: Int, spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath)
    val start = System.currentTimeMillis()
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    val model = new Word2Vec().setInputCol("token").setOutputCol("features")
      .setCBOW(0)
      .setHs(true)
      .setNegative(0)
      .setSample(0.0001)
      .setVectorSize(200)
      .setWindowSize(10)
      .setNumPartitions(executors)
      .setMaxIter(1)
      .fit(tokenizer.transform(trainData))
    val now = System.currentTimeMillis()
    val run = (now - start + 1).toDouble / 1000
    println(f"$run%.2f")
  }
  def SkipNeg(trainPath: String, executors: Int, spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath)
    val start = System.currentTimeMillis()
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    val model = new Word2Vec().setInputCol("token").setOutputCol("features")
      .setCBOW(0)
      .setHs(false)
      .setNegative(5)
      .setSample(0.0001)
      .setVectorSize(200)
      .setWindowSize(10)
      .setNumPartitions(executors)
      .setMaxIter(1)
      .fit(tokenizer.transform(trainData))
    val now = System.currentTimeMillis()
    val run = (now - start + 1).toDouble / 1000
    println(f"$run%.2f")
  }
  def CBOWHs(trainPath: String, executors: Int, spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath)
    val start = System.currentTimeMillis()
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    val model = new Word2Vec().setInputCol("token").setOutputCol("features")
      .setCBOW(1)
      .setHs(true)
      .setNegative(0)
      .setSample(0.0001)
      .setVectorSize(200)
      .setWindowSize(10)
      .setNumPartitions(executors)
      .setMaxIter(1)
      .fit(tokenizer.transform(trainData))
    val now = System.currentTimeMillis()
    val run = (now - start + 1).toDouble / 1000
    println(f"$run%.2f")
  }
  def CBOWNeg(trainPath: String, executors: Int, spark: SparkSession): Unit = {
    val trainData = spark.read.parquet(trainPath)
    val start = System.currentTimeMillis()
    val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
    val model = new Word2Vec().setInputCol("token").setOutputCol("features")
      .setCBOW(1)
      .setHs(false)
      .setNegative(5)
      .setSample(0.0001)
      .setVectorSize(200)
      .setWindowSize(10)
      .setNumPartitions(executors)
      .setMaxIter(1)
      .fit(tokenizer.transform(trainData))
    val now = System.currentTimeMillis()
    val run = (now - start + 1).toDouble / 1000
    println(f"$run%.2f")
  }
}
