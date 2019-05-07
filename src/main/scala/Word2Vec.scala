import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.nlp.Word2Vec
import org.apache.spark.sql.SparkSession

object Word2Vec extends App {
  require(args.length == 5 || args.length == 6)
  val corpusPath = args(0)
  val modelPath = args(1)
  val vectorSize = args(2).toInt
  val numPartitions = args(3).toInt
  val maxIter = args(4).toInt
  val customDic = if (args.length == 6) args(5) else "__customDic__"

  val conf = new SparkConf()
  val spark = SparkSession.builder().config(conf).getOrCreate()
  val sc = spark.sparkContext

  import spark.implicits._
  val corpus = sc.textFile(corpusPath).toDF("doc")
  val tokenizer = new Tokenizer().setInputCol("doc").setOutputCol("token")
  val w2v = new Word2Vec().setInputCol("token").setOutputCol("features")
    .setCBOW(0)
    .setCustomDic(customDic)
    .setHs(true)
    .setNegative(0)
    .setVectorSize(vectorSize)
    .setMaxIter(maxIter)
    .setNumPartitions(numPartitions)
  val model = w2v.fit(tokenizer.transform(corpus))
  model.save(modelPath)
}
