package org.apache.spark.ml.feature.nlp

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.linalg.{BLAS, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.feature.nlp
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.util.Utils

/**
  * Params for [[Word2Vec]] and [[Word2VecModel]].
  */
private[feature] trait Word2VecBase extends Params
  with HasInputCol with HasOutputCol with HasMaxIter with HasStepSize with HasSeed {

  /**
    * The path of customer's dictionary.
    * Default: __customDic__
    * @group param
    */
  final val customDic = new Param[String](
    this, "customDic", "the path of customer's dictionary")
  setDefault(customDic -> "__customDic__")
  /** @group getParam */
  def getCustomDic: String = $(customDic)

  /**
    * The value of subSampling threshold value.
    * when sample is 0, it means that subSampling is not used.
    * Default: 0.001
    * @group param
    */
  final val sample = new DoubleParam(
    this, "sample", "The value of subSampling threshold value",
    ParamValidators.gtEq(0))
  setDefault(sample -> 1e-3)
  /** @group getParam */
  def getSample: Double = $(sample)

  /**
    * The maxSize of vocab, 0 value means there is no limit.
    * Default: 0
    * @group param
    */
  final val maxVocabSize = new IntParam(
    this, "maxVocabSize", "The maxSize of vocab, 0 value means there is no limit",
    ParamValidators.gtEq(0))
  setDefault(maxVocabSize -> 0)
  /** @group getParam */
  def getMaxVocabSize: Int = $(maxVocabSize)

  /**
    * The model used to train the word vectors,
    * when this value is 0, the Skip-gram model is adopted for training
    * when this value is 1, the CBOW model is adopted for training
    * when this value is 2, the CBOW-Concat model is adopted for training
    * Default: 0
    * @group param
    */
  final val cbow = new IntParam(
    this, "cbow", "The model used to train the word vectors")
  setDefault(cbow -> 0)
  /** @group getParam */
  def getCBOW: Int= $(cbow)

  /**
    * The flag of whether use Hierarchical softmax.
    * Default: true
    * @group param
    */
  final val hs = new BooleanParam(
    this, "hs", "The flag of whether use Hierarchical softmax.")
  setDefault(hs -> true)
  /** @group getParam */
  def getHs: Boolean = $(hs)

  /**
    * The number of negative sampling.
    * when this value is 0, it means that the negative sampling is not used.
    * Default: 5
    * @group param
    */
  final val negative = new IntParam(
    this, "negative", "The number of negative sampling.",
    ParamValidators.gtEq(0))
  setDefault(negative -> 5)
  /** @group getParam */
  def getNegative: Int = $(negative)

  /**
    * The dimension of the code that you want to transform from words.
    * Default: 100
    * @group param
    */
  final val vectorSize = new IntParam(
    this, "vectorSize", "the dimension of codes after transforming from words (> 0)",
    ParamValidators.gt(0))
  setDefault(vectorSize -> 100)
  /** @group getParam */
  def getVectorSize: Int = $(vectorSize)

  /**
    * The window size (context words from [-window, window]).
    * Default: 5
    * @group param
    */
  final val windowSize = new IntParam(
    this, "windowSize", "the window size (context words from [-window, window]) (> 0)",
    ParamValidators.gt(0))
  setDefault(windowSize -> 5)
  /** @group getParam */
  def getWindowSize: Int = $(windowSize)

  /**
    * Number of partitions for sentences of words.
    * Default: 1
    * @group param
    */
  final val numPartitions = new IntParam(
    this, "numPartitions", "number of partitions for sentences of words (> 0)",
    ParamValidators.gt(0))
  setDefault(numPartitions -> 1)
  /** @group getParam */
  def getNumPartitions: Int = $(numPartitions)

  /**
    * The minimum number of times a token must appear to be included in the word2vec model's
    * vocabulary.
    * Default: 5
    * @group param
    */
  final val minCount = new IntParam(this, "minCount", "the minimum number of times a token must " +
    "appear to be included in the word2vec model's vocabulary (>= 0)", ParamValidators.gtEq(0))
  setDefault(minCount -> 5)
  /** @group getParam */
  def getMinCount: Int = $(minCount)

  /**
    * Sets the maximum length (in words) of each sentence in the input data.
    * Any sentence longer than this threshold will be divided into chunks of
    * up to `maxSentenceLength` size.
    * Default: 1000
    * @group param
    */
  final val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Maximum length " +
    "(in words) of each sentence in the input data. Any sentence longer than this threshold will " +
    "be divided into chunks up to the size (> 0)", ParamValidators.gt(0))
  setDefault(maxSentenceLength -> 1000)
  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  setDefault(stepSize -> 0.025)
  setDefault(maxIter -> 1)

  /**
    * Validate and transform the input schema.
    */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val typeCandidates = List(new ArrayType(StringType, true), new ArrayType(StringType, false))
    SchemaUtils.checkColumnTypes(schema, $(inputCol), typeCandidates)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

/**
  * Word2Vec trains a model of `Map(String, Vector)`, i.e. transforms a word into a code for further
  * natural language processing or machine learning process.
  */
final class Word2Vec(override val uid: String)
  extends Estimator[Word2VecModel] with Word2VecBase with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("word2vec"))

  /** @group setParam */
  def setSample(value: Double): this.type = set(sample, value)
  /** @group setParam */
  def setMaxVocabSize(value: Int): this.type = set(maxVocabSize, value)
  /** @group setParam */
  def setCBOW(value: Int): this.type = set(cbow, value)
  /** @group setParam */
  def setHs(value: Boolean): this.type = set(hs, value)
  /** @group setParam */
  def setNegative(value: Int): this.type = set(negative, value)
  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)
  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)
  /** @group setParam */
  def setVectorSize(value: Int): this.type = set(vectorSize, value)
  /** @group setParam */
  def setWindowSize(value: Int): this.type = set(windowSize, value)
  /** @group setParam */
  def setStepSize(value: Double): this.type = set(stepSize, value)
  /** @group setParam */
  def setNumPartitions(value: Int): this.type = set(numPartitions, value)
  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)
  /** @group setParam */
  def setMinCount(value: Int): this.type = set(minCount, value)
  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = set(maxSentenceLength, value)
  /** @group setParam */
  def setCustomDic(value: String): this.type = set(customDic,value)

  override def fit(dataset: Dataset[_]): Word2VecModel = {
    transformSchema(dataset.schema)
    val input = dataset.select($(inputCol)).rdd.map(_.getAs[Seq[String]](0))
    val word2Vectors = new nlp.Word2Vec()
      .setCBOW($(cbow))
      .setHs($(hs))
      .setNegative($(negative))
      .setSample($(sample))
      .setCustomDic($(customDic))
      .setLearningRate($(stepSize))
      .setMinCount($(minCount))
      .setNumIterations($(maxIter))
      .setNumPartitions($(numPartitions))
      .setSeed($(seed))
      .setVectorSize($(vectorSize))
      .setWindowSize($(windowSize))
      .setMaxVocabSize($(maxVocabSize))
      .setMaxSentenceLength($(maxSentenceLength))
      .fit(input)
    copyValues(new Word2VecModel(uid, word2Vectors).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
  override def copy(extra: ParamMap): Word2Vec = defaultCopy(extra)
}

object Word2Vec extends DefaultParamsReadable[Word2Vec] {
  override def load(path: String): Word2Vec = super.load(path)
}

/**
  * Model fitted by [[Word2Vec]].
  */
class Word2VecModel private[ml](override val uid: String,
                                @transient private val w2vModel: nlp.Word2VecModel)
  extends Model[Word2VecModel] with Word2VecBase with MLWritable {

  import Word2VecModel._

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)
  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
    * Returns a dataframe with two fields, "word" and "vector", with "word" being a String and
    * and the "vector" the DenseVector that it is mapped to.
    */
  @transient lazy val getVectors: DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    val wordVec = w2vModel.getVectors
      .mapValues(vec => Vectors.dense(vec.toArray))
    spark.createDataFrame(wordVec.toSeq).toDF("word", "vector")
  }
  /**
    * Returns a dataframe with two fields, "word" and "count", with "word" being a String and
    * and the "count" the frequency that it is mapped to.
    */
  @transient lazy val getVocab: DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    val wordCount = w2vModel.getVocab
    spark.createDataFrame(wordCount.toSeq).toDF("word", "count")
  }
  /**
    * Find "num" number of words closest in similarity to the given word, not
    * including the word itself.
    * @return a dataframe with columns "word" and "similarity" of the word and the cosine
    * similarities between the synonyms and the given word.
    */
  def findSynonyms(word: String, num: Int): DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    spark.createDataFrame(findSynonymsArray(word, num)).toDF("word", "similarity")
  }
  /**
    * Find "num" number of words whose vector representation is most similar to the supplied vector.
    * If the supplied vector is the vector representation of a word in the model's vocabulary,
    * that word will be in the results.
    * @return a dataframe with columns "word" and "similarity" of the word and the cosine
    * similarities between the synonyms and the given word vector.
    */
  def findSynonyms(vec: Vector, num: Int): DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    spark.createDataFrame(findSynonymsArray(vec, num)).toDF("word", "similarity")
  }
  /**
    * Find "num" number of words whose vector representation is most similar to the supplied vector.
    * If the supplied vector is the vector representation of a word in the model's vocabulary,
    * that word will be in the results.
    * @return an array of the words and the cosine similarities between the synonyms given
    * word vector.
    */
  def findSynonymsArray(vec: Vector, num: Int): Array[(String, Double)] = {
    w2vModel.findSynonyms(vec, num)
  }
  /**
    * Find "num" number of words closest in similarity to the given word, not
    * including the word itself.
    * @return an array of the words and the cosine similarities between the synonyms given
    * word vector.
    */
  def findSynonymsArray(word: String, num: Int): Array[(String, Double)] = {
    w2vModel.findSynonyms(word, num)
  }

  /**
    * Transform a sentence column to a vector column to represent the whole sentence. The transform
    * is performed by averaging all word vectors it contains.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val vectors = w2vModel.getVectors
      .mapValues(vec => Vectors.dense(vec.toArray))
      .map(identity)
    val bVectors = dataset.sparkSession.sparkContext.broadcast(vectors)
    val vecSize = $(vectorSize)
    val word2Vec = udf { sentence: Seq[String] =>
      if (sentence.isEmpty) {
        Vectors.sparse(vecSize, Array.empty[Int], Array.empty[Double])
      } else {
        val sum = Vectors.zeros(vecSize)
        sentence.foreach { word =>
          bVectors.value.get(word).foreach { v =>
            BLAS.axpy(1.0, v, sum)
          }
        }
        BLAS.scal(1.0 / sentence.size, sum)
        sum
      }
    }
    dataset.withColumn($(outputCol), word2Vec(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
  override def copy(extra: ParamMap): Word2VecModel = {
    val copied = new Word2VecModel(uid, w2vModel)
    copyValues(copied, extra).setParent(parent)
  }
  override def write: MLWriter = new Word2VecModelWriter(this)

}

object Word2VecModel extends MLReadable[Word2VecModel] {
  private case class Data(word: String, vector: Array[Float], count: Int)

  private[Word2VecModel]
  class Word2VecModelWriter(instance: Word2VecModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession
      import spark.implicits._

      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val wordVectors = instance.w2vModel.getVectors
        .mapValues(vec => vec.toArray.map(_.toFloat)).toSeq
      val wordVectorsRDD = sc.parallelize(wordVectors)
      val wordCount = instance.w2vModel.getVocab.toSeq
      val wordCountRDD = sc.parallelize(wordCount)

      val dataPath = new Path(path, "data").toString
      val bufferSizeInBytes = Utils.byteStringAsBytes(
        sc.conf.get("spark.kryoserializer.buffer.max", "64m"))
      val numPartitions = Word2VecModelWriter.calculateNumberOfPartitions(
        bufferSizeInBytes, instance.w2vModel.wordIndex.size, instance.getVectorSize)

      wordVectorsRDD.join(wordCountRDD)
        .repartition(numPartitions)
        .map{case(word, (vec, count)) => Data(word, vec, count)}
        .toDF()
        .write
        .parquet(dataPath)
    }
  }

  private[feature]
  object Word2VecModelWriter {
    /**
      * Calculate the number of partitions to use in saving the model.
      * [SPARK-11994] - We want to partition the model in partitions smaller than
      * spark.kryoserializer.buffer.max
      * @param bufferSizeInBytes  Set to spark.kryoserializer.buffer.max
      * @param numWords  Vocab size
      * @param vectorSize  Vector length for each word
      */
    def calculateNumberOfPartitions(
                                     bufferSizeInBytes: Long,
                                     numWords: Int,
                                     vectorSize: Int): Int = {
      val floatSize = 4L  // Use Long to help avoid overflow
      val averageWordSize = 15
      // Calculate the approximate size of the model.
      // Assuming an average word size of 15 bytes, the formula is:
      // (floatSize * vectorSize + 15) * numWords
      val approximateSizeInBytes = (floatSize * vectorSize + averageWordSize) * numWords
      val numPartitions = (approximateSizeInBytes / bufferSizeInBytes) + 1
      require(numPartitions < 10e8, s"Word2VecModel calculated that it needs $numPartitions " +
        s"partitions to save this model, which is too large.  Try increasing " +
        s"spark.kryoserializer.buffer.max so that Word2VecModel can use fewer partitions.")
      numPartitions.toInt
    }
  }

  private class Word2VecModelReader extends MLReader[Word2VecModel] {

    private val className = classOf[Word2VecModel].getName

    override def load(path: String): Word2VecModel = {
      val spark = sparkSession
      import spark.implicits._
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = spark.read.parquet(dataPath).as[Data].collect()

      val wordVectorsMap = data
        .map(wordVector => (wordVector.word, wordVector.vector))
        .toMap
      val wordCount = data
        .map(wordVector => (wordVector.word, wordVector.count))
        .toMap

      val model = new Word2VecModel(metadata.uid, new nlp.Word2VecModel(wordVectorsMap, wordCount))
      metadata.getAndSetParams(model)
      model
    }
  }

  override def read: MLReader[Word2VecModel] = new Word2VecModelReader
  override def load(path: String): Word2VecModel = super.load(path)
}
