package org.apache.spark.ml.feature.nlp

import org.apache.spark.ml.linalg.{VectorUDT, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.feature.nlp
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}


private[feature] trait Doc2VecBase extends Params
  with HasInputCol with HasOutputCol with HasMaxIter with HasStepSize with HasSeed {

  final val decay = new DoubleParam(
    this, "decay", "alpha decay",
    ParamValidators.inRange(0d, 1d, false, true))
  setDefault(decay -> 1.0d)
  def getDecay: Double = $(decay)

  final val window = new IntParam(
    this, "window", "window",
    ParamValidators.gtEq(0))
  setDefault(window -> 5)
  def getWindow: Int = $(window)

  final val docIdCol = new Param[String](
    this, "docIdCol", "this id of each document")
  setDefault(docIdCol -> "id")
  def getIdCol: String = $(docIdCol)

  final val customDic = new Param[String](
    this, "customDic", "the path of custom dictionary: <word frequency>")
  setDefault(customDic -> "__customDic__")
  def getCustomDic: String = $(customDic)

  final val sample = new DoubleParam(
    this, "sample", "sample use for sub-Sampling",
    ParamValidators.gtEq(0))
  setDefault(sample -> 1e-4)
  def getSample: Double = $(sample)

  final val maxVocabSize = new IntParam(
    this, "maxVocabSize", "set the max VocabSize, 0 means vocabSize is no limited",
    ParamValidators.gtEq(0))
  setDefault(maxVocabSize -> 0)
  def getMaxVocabSize: Int = $(maxVocabSize)

  final val vectorSize = new IntParam(
    this, "vectorSize", "the dimension of codes after transforming from words (> 0)",
    ParamValidators.gt(0))
  setDefault(vectorSize -> 100)
  def getVectorSize: Int = $(vectorSize)

  final val numPartitions = new IntParam(
    this, "numPartitions", "number of partitions for sentences of words (> 0)",
    ParamValidators.gt(0))
  setDefault(numPartitions -> 1)
  def getNumPartitions: Int = $(numPartitions)

  final val minCount = new IntParam(this, "minCount", "the minimum number of times a token must " +
    "appear to be included in the word2vec model's vocabulary (>= 0)", ParamValidators.gtEq(0))
  setDefault(minCount -> 5)
  def getMinCount: Int = $(minCount)

  setDefault(stepSize -> 0.025)
  setDefault(maxIter -> 1)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val typeCandidates = List(new ArrayType(StringType, true), new ArrayType(StringType, false))
    SchemaUtils.checkColumnTypes(schema, $(inputCol), typeCandidates)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

final class Doc2Vec(override val uid: String)
  extends Estimator[Doc2VecModel] with Doc2VecBase with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("doc2vec"))

  def setWindow(value: Int): this.type = set(window, value)
  def setDocIdCol(value: String): this.type = set(docIdCol, value)
  def setCustomDic(value: String): this.type = set(customDic,value)
  def setSample(value: Double): this.type = set(sample, value)
  def setMaxVocabSize(value: Int): this.type = set(maxVocabSize, value)
  def setDecay(value: Double): this.type = set(decay, value)
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setVectorSize(value: Int): this.type = set(vectorSize, value)
  def setStepSize(value: Double): this.type = set(stepSize, value)
  def setNumPartitions(value: Int): this.type = set(numPartitions, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setSeed(value: Long): this.type = set(seed, value)
  def setMinCount(value: Int): this.type = set(minCount, value)

  override def fit(dataset: Dataset[_]): Doc2VecModel = {
    transformSchema(dataset.schema)
    val input = dataset.select($(docIdCol), $(inputCol)).rdd
      .map(row => (row.getInt(0), row.getAs[Seq[String]](1)))
    val doc2VectorModel = new nlp.Doc2Vec()
      .setWindow($(window))
      .setSample($(sample))
      .setDecay($(decay))
      .setCustomDic($(customDic))
      .setLearningRate($(stepSize))
      .setMinCount($(minCount))
      .setNumIterations($(maxIter))
      .setNumPartitions($(numPartitions))
      .setSeed($(seed))
      .setVectorSize($(vectorSize))
      .setMaxVocabSize($(maxVocabSize))
      .fit(input)
    copyValues(new Doc2VecModel(uid, doc2VectorModel).setParent(this))
  }
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
  override def copy(extra: ParamMap): Doc2Vec = defaultCopy(extra)
}

object Doc2Vec extends DefaultParamsReadable[Doc2Vec] {
  override def load(path: String): Doc2Vec = super.load(path)
}

class Doc2VecModel private[ml](override val uid: String,
                                @transient private val docModel: nlp.Doc2VecModel)
  extends Model[Doc2VecModel] with Doc2VecBase {

  @transient lazy val getVectors: DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    val wordVec = docModel.getVectors
      .mapValues(vec => Vectors.dense(vec.toArray))
    spark.createDataFrame(wordVec.toSeq).toDF("id", "vector")
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    val bcModel = dataset.sparkSession.sparkContext.broadcast(docModel)
    val doc2vec = udf { doc: Seq[String] =>
      val vec = bcModel.value.transform(doc)
      Vectors.dense(vec.toArray)
    }
    dataset.withColumn($(outputCol), doc2vec(col($(inputCol))))
  }
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
  override def copy(extra: ParamMap): Doc2VecModel = {
    val copied = new Doc2VecModel(uid, docModel)
    copyValues(copied, extra).setParent(parent)
  }
}

