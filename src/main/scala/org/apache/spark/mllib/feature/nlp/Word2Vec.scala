package org.apache.spark.mllib.feature.nlp

import java.lang.{Iterable => JavaIterable}

import org.apache.hadoop.fs.Path
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd._
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.util.{BoundedPriorityQueue, Utils}
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.control.Breaks._

/**
  *  Entry in vocabulary
  */
private case class VocabWord(
  var word: String,
  var cn: Int,
  var point: Array[Int],
  var code: Array[Int],
  var codeLen: Int) extends Serializable

/**
  * Word2Vec creates vector representation of words in a text corpus.
  * The algorithm first constructs a vocabulary from the corpus
  * and then learns vector representation of words in the vocabulary.
  * The vector representation can be used as features in
  * natural language processing and machine learning algorithms.
  */
class Word2Vec extends Serializable with Logging {

  private var vectorSize = 100
  private var learningRate = 0.025
  private var numPartitions = 1
  private var numIterations = 1
  private var seed = Utils.random.nextLong()
  private var minCount = 5
  private var maxSentenceLength = 1000
  private var window = 5

  private var cbow = 0
  private var hs = true
  private var negative = 0
  private var sample = 1e-3
  private var maxVocabSize = 0
  private var customDic = "__customDic__"

  private var useCustomDic = false
  private var minReduce = 1

  /**
    * sets the model flag(default: 0).
    * when cbow is 0, the Skip-gram model is adopted for training.
    * when cbow is 1, the CBOW model is adopted for training.
    * when cbow is 2, the CBOW-Concat model is adopted for training.
    */
  def setCBOW(cbow: Int): this.type = {
    require(cbow >=0 && cbow < 3, "cbow must be one of [0, 1 ,2]")
    this.cbow = cbow
    this
  }
  /**
    * sets the flag of whether use Hierarchical softmax(default: true).
    */
  def setHs(hs: Boolean): this.type = {
    this.hs = hs
    this
  }
  /**
    * set the value of negative sampling (default: 5).
    * when this value is 0, it means that the negative sampling is not used.
    */
  def setNegative(negative: Int): this.type = {
    require(negative >= 0,
      s"Number of negative sample must be nonnegative but got $negative")
    this.negative = negative
    this
  }
  /**
    * set the value of subSampling threshold value(default: 0.001).
    * when sample is 0, it means that subSampling is not used.
    */
  def setSample(sample: Double): this.type = {
    require(sample >= 0, s"sample must be nonnegative but got $sample")
    this.sample = sample
    this
  }
  /**
    * set the maxSize of vocab, 0 value means there is no limit(default: 0).
    */
  def setMaxVocabSize(size: Int): this.type = {
    require(sample>=0, s"maxVocabSize must be nonnegative but got $sample")
    this.maxVocabSize = size
    this
  }
  /**
    * set the path of customer's dictionary.
    */
  def setCustomDic(dic: String): this.type = {
    this.customDic = dic
    this
  }
  /**
    * Sets initial learning rate (default: 0.025).
    */
  def setMaxSentenceLength(maxSentenceLength: Int): this.type = {
    require(maxSentenceLength > 0,
      s"Maximum length of sentences must be positive but got $maxSentenceLength")
    this.maxSentenceLength = maxSentenceLength
    this
  }
  /**
    * Sets vector size (default: 100).
    */
  def setVectorSize(vectorSize: Int): this.type = {
    require(vectorSize > 0,
      s"vector size must be positive but got $vectorSize")
    this.vectorSize = vectorSize
    this
  }
  /**
    * Sets initial learning rate (default: 0.025).
    */
  def setLearningRate(learningRate: Double): this.type = {
    require(learningRate > 0,
      s"Initial learning rate must be positive but got $learningRate")
    this.learningRate = learningRate
    this
  }
  /**
    * Sets number of partitions (default: 1). Use a small number for accuracy.
    */
  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0,
      s"Number of partitions must be positive but got $numPartitions")
    this.numPartitions = numPartitions
    this
  }
  /**
    * Sets number of iterations (default: 1), which should be smaller than or equal to number of
    * partitions.
    */
  def setNumIterations(numIterations: Int): this.type = {
    require(numIterations >= 0,
      s"Number of iterations must be nonnegative but got $numIterations")
    this.numIterations = numIterations
    this
  }
  /**
    * Sets random seed (default: a random long integer).
    */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }
  /**
    * Sets the window of words (default: 5)
    */
  def setWindowSize(window: Int): this.type = {
    require(window > 0,
      s"Window of words must be positive but got $window")
    this.window = window
    this
  }
  /**
    * Sets minCount, the minimum number of times a token must appear to be included in the word2vec
    * model's vocabulary (default: 5).
    */
  def setMinCount(minCount: Int): this.type = {
    require(minCount >= 0,
      s"Minimum number of times must be nonnegative but got $minCount")
    this.minCount = minCount
    this
  }

  private val EXP_TABLE_SIZE = 6000
  private val MAX_EXP = 6
  private val MAX_CODE_LENGTH = 40
  private val NEG_TABLE_SIZE = 1e10.toInt

  private var trainWordsCount = 0L
  private var concatVectorSize = 0
  private var vocabSize = 0
  @transient private var vocab: Array[VocabWord] = null
  @transient private var vocabHash = mutable.HashMap.empty[String, Int]

  private def learnVocab[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    val sc = dataset.sparkContext
    if(useCustomDic) {
      val dic = sc.textFile(customDic)
        .map(_.split(" "))
        .map(x => (x(0), x(1).toInt))
      vocab = dic.filter(_._2 >= minCount)
        .map(x => VocabWord(x._1, x._2, null, null, 0))
        .collect()
        .sortWith((a, b) => a.cn > b.cn)
    } else {
      vocab = dataset.flatMap(x => x)
        .map(w => (w, 1))
        .reduceByKey(_ + _)
        .filter(_._2 >= minCount)
        .map(x => VocabWord(x._1, x._2, null, null, 0))
        .collect()
        .sortWith((a, b) => a.cn > b.cn)
    }

    vocabSize = vocab.length
    if(maxVocabSize > 0) reduceVocab()

    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    for (i <- 0 until vocabSize) {
      vocabHash += vocab(i).word -> i
      trainWordsCount += vocab(i).cn
    }

    logInfo(s"vocabSize = $vocabSize, trainWordsCount / Iter = $trainWordsCount")
  }
  private def reduceVocab(): Unit =  {
    minReduce = minCount + 1
    while(vocabSize > maxVocabSize) {
      vocab = vocab.filter(_.cn > minReduce).sortWith((a, b) => a.cn > b.cn)
      vocabSize = vocab.length
      minReduce += 1
    }
  }
  private def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      val tmp = math.exp(-(2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (1 / (1 + tmp)).toFloat
      i += 1
    }
    expTable
  }
  private def createNegTable(): Array[Int] = {
    var trainWordsPow = 0d
    val power = 0.75
    val table = new Array[Int](NEG_TABLE_SIZE)
    vocab.foreach(word => trainWordsPow += math.pow(word.cn, power))
    var i = 0
    var d = math.pow(vocab(i).cn, power) / trainWordsPow

    for (j <- 0 until NEG_TABLE_SIZE) {
      table(j) = i

      if( j.toDouble / NEG_TABLE_SIZE > d) {
        i += 1
        d += math.pow(vocab(i).cn, power) / trainWordsPow
      }
      if( i >= vocabSize) i = vocabSize-1

    }
    table
  }
  private def createBinaryTree(): Unit = {
    val count = new Array[Long](vocabSize * 2 + 1)
    val binary = new Array[Int](vocabSize * 2 + 1)
    val parentNode = new Array[Int](vocabSize * 2 + 1)
    val code = new Array[Int](MAX_CODE_LENGTH)
    val point = new Array[Int](MAX_CODE_LENGTH)
    var a = 0
    while (a < vocabSize) {
      count(a) = vocab(a).cn
      a += 1
    }
    while (a < 2 * vocabSize) {
      count(a) = 1e9.toInt
      a += 1
    }
    var pos1 = vocabSize - 1
    var pos2 = vocabSize

    var min1i = 0
    var min2i = 0

    a = 0
    while (a < vocabSize - 1) {
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min1i = pos1
          pos1 -= 1
        } else {
          min1i = pos2
          pos2 += 1
        }
      } else {
        min1i = pos2
        pos2 += 1
      }
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min2i = pos1
          pos1 -= 1
        } else {
          min2i = pos2
          pos2 += 1
        }
      } else {
        min2i = pos2
        pos2 += 1
      }
      count(vocabSize + a) = count(min1i) + count(min2i)
      parentNode(min1i) = vocabSize + a
      parentNode(min2i) = vocabSize + a
      binary(min2i) = 1
      a += 1
    }
    // Now assign binary code to each vocabulary word
    var i = 0
    a = 0
    while (a < vocabSize) {
      vocab(a).code = new Array[Int](MAX_CODE_LENGTH)
      vocab(a).point = new Array[Int](MAX_CODE_LENGTH)
      var b = a
      i = 0
      while (b != vocabSize * 2 - 2) {
        code(i) = binary(b)
        point(i) = b
        i += 1
        b = parentNode(b)
      }
      vocab(a).codeLen = i
      vocab(a).point(0) = vocabSize - 2
      b = 0
      while (b < i) {
        vocab(a).code(i - b - 1) = code(b)
        vocab(a).point(i - b) = point(b) - vocabSize
        b += 1
      }
      a += 1
    }
  }

  /**
    * Computes the vector representation of each word in vocabulary.
    * @param dataset an RDD of sentences,
    *                each sentence is expressed as an iterable collection of words
    * @return a Word2VecModel
    */
  def fit[S <: Iterable[String]](dataset: RDD[S]): Word2VecModel = {
    if(customDic != "__customDic__") useCustomDic = true
    val sc = dataset.context

    learnVocab(dataset)
    if(hs) createBinaryTree()

    val expTable = sc.broadcast(createExpTable())
    val negTable = if (negative > 0) sc.broadcast(createNegTable()) else null
    val bcVocab = sc.broadcast(vocab)
    val bcVocabHash = sc.broadcast(vocabHash)
    try {
      doFit(dataset, sc, expTable, negTable, bcVocab, bcVocabHash)
    } finally {
      expTable.destroy(blocking = false)
      bcVocab.destroy(blocking = false)
      bcVocabHash.destroy(blocking = false)
      if(negative > 0) negTable.destroy(blocking = false)
    }
  }
  /**
    * Computes the vector representation of each word in vocabulary (Java version).
    * @param dataset a JavaRDD of words
    * @return a Word2VecModel
    */
  def fit[S <: JavaIterable[String]](dataset: JavaRDD[S]): Word2VecModel = {
    fit(dataset.rdd.map(_.asScala))
  }

  private def doFit[S <: Iterable[String]](
    dataset: RDD[S],
    sc: SparkContext,
    expTable: Broadcast[Array[Float]],
    negTable: Broadcast[Array[Int]],
    bcVocab: Broadcast[Array[VocabWord]],
    bcVocabHash: Broadcast[mutable.HashMap[String, Int]]): Word2VecModel = {
    val sentences: RDD[Array[Int]] = dataset.mapPartitions { sentenceIter =>
      sentenceIter.flatMap { sentence =>
        val wordIndexes = sentence.flatMap(bcVocabHash.value.get)
        wordIndexes.grouped(maxSentenceLength).map(_.toArray)
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()
    val initRandom = new XORShiftRandom(seed)
    concatVectorSize = vectorSize * (window * 2 + 1)

    if (vocabSize.toLong * vectorSize >= Int.MaxValue) {
      throw new RuntimeException("Please increase minCount or decrease vectorSize in Word2Vec" +
        " to avoid an OOM. You are highly recommended to make your vocabSize*vectorSize, " +
        "which is " + vocabSize + "*" + vectorSize + " for now, less than `Int.MaxValue`.")
    }

    val syn0Global = Array.fill[Float](vocabSize * vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    val syn1Global = if (hs) {
      if(cbow == 2) new Array[Float](vocabSize * concatVectorSize)
      else new Array[Float](vocabSize * vectorSize)
    } else null
    val syn1negGlobal = if (negative > 0) {
      if(cbow == 2) new Array[Float](vocabSize * concatVectorSize)
      else new Array[Float](vocabSize * vectorSize)
    } else null

    val totalWordsCounts = numIterations * trainWordsCount + 1
    var alpha = learningRate
    val startTime = System.currentTimeMillis()

    for (k <- 1 to numIterations) {
      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)
      val bcSyn1negGlobal = sc.broadcast(syn1negGlobal)
      val numWordsProcessedInPreviousIterations = (k - 1) * trainWordsCount

      val partial = newSentences.mapPartitionsWithIndex { case (idx, iter) =>
        val random = new XORShiftRandom(seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))
        val syn0Modify = new Array[Int](vocabSize)
        val syn1Modify = if(hs) new Array[Int](vocabSize) else null
        val syn1negModify = if(negative > 0) new Array[Int](vocabSize) else null

        val neu1 = if(cbow == 2) new Array[Float](concatVectorSize)
          else new Array[Float](vectorSize)
        val neu1e = if(cbow == 2) new Array[Float](concatVectorSize)
          else new Array[Float](vectorSize)

        val model = iter.foldLeft((bcSyn0Global.value, bcSyn1Global.value, bcSyn1negGlobal.value, 0L, 0L)) {
          case ((syn0, syn1, syn1neg, lastWordCount, wordCount), sentence) =>
            var lwc = lastWordCount
            var wc = wordCount
            if (wordCount - lastWordCount > 10000) {
              lwc = wordCount
              val wordCountActual = numPartitions * wordCount.toDouble + numWordsProcessedInPreviousIterations
              alpha = learningRate * (1 - wordCountActual / totalWordsCounts)
              if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001

              val runTime = (System.currentTimeMillis() - startTime + 1).toDouble / 1000
              val progress = wordCountActual / totalWordsCounts * 100
              val wordsSec = (wordCountActual / runTime).toInt
              logInfo(s"wordCount = ${wordCountActual.toInt}, " + f"alpha = $alpha%.10f, " +
                f"runtime = $runTime%.2f S, " + f"progress = $progress%.2f%%, " + s"words/S = $wordsSec")
            }
            wc += sentence.length
            for (pos <- sentence.indices) {
              breakable {
                val word = sentence(pos)
                if (sample > 0) {
                  val ran = (math.sqrt(bcVocab.value(word).cn / (sample * trainWordsCount)) + 1) *
                    (sample * trainWordsCount) / bcVocab.value(word).cn
                  if (ran < random.nextDouble()) break()
                }
                val b = random.nextInt(window)
                // Train Skip-gram
                if (cbow == 0) {
                  for (i <- b - window to window - b if i != 0) {
                    val c = pos + i
                    if (c >= 0 && c < sentence.length) {
                      val lastWord = sentence(c)
                      val l1 = lastWord * vectorSize
                      blas.sscal(vectorSize, 0f, neu1e, 1)
                      // Hierarchical softMax
                      if (hs) for (i <- 0 until bcVocab.value(word).codeLen) {
                        val inner = bcVocab.value(word).point(i)
                        val l2 = inner * vectorSize
                        val f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                        val sig = if (f > MAX_EXP) 1 else if (f < -MAX_EXP) 0 else {
                            val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                            expTable.value(ind)
                          }
                        val g = ((1 - bcVocab.value(word).code(i) - sig) * alpha).toFloat
                        blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                        blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1)
                        syn1Modify(inner) += 1
                      }
                      // Negative Sampling
                      if (negative > 0) for (i <- 0 to negative) {
                        val label = if (i == 0) 1 else 0
                        val target = if (i == 0) word else {
                          var tmp = 0
                          do {
                            tmp = negTable.value(random.nextInt(NEG_TABLE_SIZE))
                          } while (tmp == word)
                          tmp
                        }
                        val l2 = target * vectorSize
                        val f = blas.sdot(vectorSize, syn0, l1, 1, syn1neg, l2, 1)
                        val sig = if (f > MAX_EXP) 1 else if (f < -MAX_EXP) 0 else {
                          val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                          expTable.value(ind)
                        }
                        val g = ((label - sig) * alpha).toFloat
                        blas.saxpy(vectorSize, g, syn1neg, l2, 1, neu1e, 0, 1)
                        blas.saxpy(vectorSize, g, syn0, l1, 1, syn1neg, l2, 1)
                        syn1negModify(target) += 1
                      }
                      blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1)
                      syn0Modify(lastWord) += 1
                    }
                  }
                }
                // Train CBOW without Concat vector
                if (cbow == 1) {
                  var cw = 0
                  blas.sscal(vectorSize, 0f, neu1, 1)
                  for (i <- b - window to window - b if i != 0) {
                    val c = pos + i
                    if (c >= 0 && c < sentence.length) {
                      val lastWord = sentence(c)
                      val l1 = lastWord * vectorSize
                      blas.saxpy(vectorSize, 1.0f, syn0, l1, 1, neu1, 0, 1)
                      cw += 1
                    }
                  }
                  if (cw > 0) {
                    blas.sscal(vectorSize, 0f, neu1e, 1)
                    blas.sscal(vectorSize, 1 / cw.toFloat, neu1, 1)
                    // Hierarchical softMax
                    if (hs) for (i <- 0 until bcVocab.value(word).codeLen) {
                      val inner = bcVocab.value(word).point(i)
                      val l2 = inner * vectorSize
                      val f = blas.sdot(vectorSize, neu1, 0, 1, syn1, l2, 1)
                      val sig = if (f > MAX_EXP) 1 else if (f < -MAX_EXP) 0 else {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                        expTable.value(ind)
                      }
                      val g = ((1 - bcVocab.value(word).code(i) - sig) * alpha).toFloat
                      blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                      blas.saxpy(vectorSize, g, neu1, 0, 1, syn1, l2, 1)
                      syn1Modify(inner) += 1
                    }
                    // Negative Sampling
                    if (negative > 0) for (i <- 0 to negative) {
                      val label = if (i == 0) 1 else 0
                      val target = if (i == 0) word else {
                        var tmp = 0
                        do {
                          tmp = negTable.value(random.nextInt(NEG_TABLE_SIZE))
                        } while (tmp == word)
                        tmp
                      }
                      val l2 = target * vectorSize
                      val f = blas.sdot(vectorSize, neu1, 0, 1, syn1neg, l2, 1)
                      val sig = if (f > MAX_EXP) 1 else if (f < -MAX_EXP) 0 else {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                        expTable.value(ind)
                      }
                      val g = ((label - sig) * alpha).toFloat
                      blas.saxpy(vectorSize, g, syn1neg, l2, 1, neu1e, 0, 1)
                      blas.saxpy(vectorSize, g, neu1, 0, 1, syn1neg, l2, 1)
                      syn1negModify(target) += 1
                    }
                    for (i <- b - window to window - b if i != 0) {
                      val c = pos + i
                      if (c >= 0 && c < sentence.length) {
                        val lastWord = sentence(c)
                        val l1 = lastWord * vectorSize
                        blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1)
                        syn0Modify(lastWord) += 1
                      }
                    }
                  }
                }
                // Train CBOW with Concat vector
                if (cbow == 2) {
                  var cw = 0
                  blas.sscal(concatVectorSize, 0f, neu1, 1)
                  for(i <- b - window to window - b) {
                    val c = pos + i
                    val index = i + window - b
                    if (c >= 0 && c < sentence.length && c != pos) {
                      val lastWord = sentence(c)
                      val l1 = lastWord * vectorSize
                      Array.copy(syn0, l1, neu1, index * vectorSize, vectorSize)
                      cw += 1
                    }
                  }
                  if (cw > 0) {
                    blas.sscal(concatVectorSize, 0f, neu1e, 1)
                    // Hierarchical softMax
                    if (hs) for (i <- 0 until bcVocab.value(word).codeLen) {
                      val inner = bcVocab.value(word).point(i)
                      val l2 = inner * concatVectorSize
                      val f = blas.sdot(concatVectorSize, neu1, 0, 1, syn1, l2, 1)
                      val sig = if (f > MAX_EXP) 1 else if (f < -MAX_EXP) 0 else {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                        expTable.value(ind)
                      }
                      val g = ((1 - bcVocab.value(word).code(i) - sig) * alpha).toFloat
                      blas.saxpy(concatVectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                      blas.saxpy(concatVectorSize, g, neu1, 0, 1, syn1, l2, 1)
                      syn1Modify(inner) += 1
                    }
                    // Negative Sampling
                    if (negative > 0) for (i <- 0 to negative) {
                      val label = if (i == 0) 1 else 0
                      val target = if (i == 0) word else {
                        var tmp = 0
                        do {
                          tmp = negTable.value(random.nextInt(NEG_TABLE_SIZE))
                        } while (tmp == word)
                        tmp
                      }
                      val l2 = target * concatVectorSize
                      val f = blas.sdot(concatVectorSize, neu1, 0, 1, syn1neg, l2, 1)
                      val sig = if (f > MAX_EXP) 1 else if (f < -MAX_EXP) 0 else {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                        expTable.value(ind)
                      }
                      val g = ((label - sig) * alpha).toFloat
                      blas.saxpy(concatVectorSize, g, syn1neg, l2, 1, neu1e, 0, 1)
                      blas.saxpy(concatVectorSize, g, neu1, 0, 1, syn1neg, l2, 1)
                      syn1negModify(target) += 1
                    }
                    for(i <- b - window to window - b) {
                      val c = pos + i
                      val index = i + window - b
                      if (c >= 0 && c < sentence.length && c != pos) {
                        val lastWord = sentence(c)
                        val l1 = lastWord * vectorSize
                        blas.saxpy(vectorSize, 1.0f, neu1e, index * vectorSize, 1, syn0, l1, 1)
                        syn0Modify(lastWord) += 1
                      }
                    }
                  }
                }
              }
            }
            (syn0, syn1, syn1neg, lwc, wc)
        }
        val syn0Local = model._1
        val syn1Local = model._2
        val syn1negLocal = model._3

        // Only output modified vectors.
        Iterator.tabulate(vocabSize) { index =>
          if (syn0Modify(index) > 0) {
            Some((index,
              (syn0Local.slice(index * vectorSize, (index + 1) * vectorSize), 1)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (hs && numIterations > 1 && syn1Modify(index) > 0) {
            if(cbow == 2) {
              Some((index + vocabSize,
                (syn1Local.slice(index * concatVectorSize, (index + 1) * concatVectorSize), 1)))
            } else {
              Some((index + vocabSize,
                (syn1Local.slice(index * vectorSize, (index + 1) * vectorSize), 1)))
            }
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (negative > 0 && numIterations > 1 && syn1negModify(index) > 0) {
            if(cbow == 2) {
              Some((index + vocabSize * 2,
                (syn1negLocal.slice(index * concatVectorSize, (index + 1) * concatVectorSize), 1)))
            } else {
              Some((index + vocabSize * 2,
                (syn1negLocal.slice(index * vectorSize, (index + 1) * vectorSize), 1)))
            }
          } else {
            None
          }
        }.flatten
      }

      val synAgg = partial.reduceByKey { case ((v1, n1), (v2, n2)) =>
        if(v1.length == vectorSize) {
          blas.saxpy(vectorSize, 1.0f, v2, 1, v1, 1)
        } else {
          blas.saxpy(concatVectorSize, 1.0f, v2, 1, v1, 1)
        }
        (v1, n1 + n2)
      }.mapValues { case(v, n) =>
        if(v.length == vectorSize) {
          blas.sscal(vectorSize, 1 / n.toFloat, v, 1)
        } else {
          blas.sscal(concatVectorSize, 1 / n.toFloat, v, 1)
        }
        v
      }.collect()

      for (word <- synAgg) {
        val index = word._1
        val vec = word._2
        if (index < vocabSize) {
          Array.copy(vec, 0, syn0Global, index * vectorSize, vectorSize)
        } else if (index < vocabSize * 2) {
          if(cbow == 2) {
            Array.copy(vec, 0, syn1Global, (index - vocabSize) * concatVectorSize, concatVectorSize)
          } else {
            Array.copy(vec, 0, syn1Global, (index - vocabSize) * vectorSize, vectorSize)
          }
        } else {
          if(cbow == 2) {
            Array.copy(vec, 0, syn1negGlobal, (index - vocabSize * 2) * concatVectorSize, concatVectorSize)
          } else {
            Array.copy(vec, 0, syn1negGlobal, (index - vocabSize * 2) * vectorSize, vectorSize)
          }
        }
      }
      bcSyn0Global.destroy(false)
      bcSyn1Global.destroy(false)
      bcSyn1negGlobal.destroy(false)
    }
    newSentences.unpersist()
    val wordsCount = vocab.map(x => (x.word, x.cn)).toMap
    new Word2VecModel(vocabHash.toMap, syn0Global, wordsCount)
  }
}
/**
  * Word2Vec model
  * @param wordIndex maps each word to an index, which can retrieve the corresponding
  *                  vector from wordVectors
  * @param wordVectors array of length numWords * vectorSize, vector corresponding
  *                    to the word mapped with index i can be retrieved by the slice
  *                    (i * vectorSize, i * vectorSize + vectorSize)
  * @param wordCount maps each word to an frequency
  */
class Word2VecModel private[spark](
  private[spark] val wordIndex: Map[String, Int],
  private[spark] val wordVectors: Array[Float],
  private[spark] val wordCount: Map[String, Int]) extends Serializable with Saveable {

  def this(model: Map[String, Array[Float]], wordCount: Map[String, Int]) = {
    this(Word2VecModel.buildWordIndex(model), Word2VecModel.buildWordVectors(model), wordCount)
  }

  private val numWords = wordIndex.size
  // vectorSize: Dimension of each word's vector.
  private val vectorSize = wordVectors.length / numWords
  // wordList: Ordered list of words obtained from wordIndex.
  private val wordList: Array[String] = {
    val (wl, _) = wordIndex.toSeq.sortBy(_._2).unzip
    wl.toArray
  }
  // wordVecNorms: Array of length numWords, each value being the Euclidean norm
  //               of the wordVector.
  private val wordVecNorms: Array[Float] = {
    val wordVecNorms = new Array[Float](numWords)
    var i = 0
    while (i < numWords) {
      val vec = wordVectors.slice(i * vectorSize, i * vectorSize + vectorSize)
      wordVecNorms(i) = blas.snrm2(vectorSize, vec, 1)
      i += 1
    }
    wordVecNorms
  }

  /**
    * Transforms a word to its vector representation
    * @param word a word
    * @return vector representation of word
    */
  def transform(word: String): Vector = {
    wordIndex.get(word) match {
      case Some(ind) =>
        val vec = wordVectors.slice(ind * vectorSize, ind * vectorSize + vectorSize)
        Vectors.dense(vec.map(_.toDouble))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }
  /**
    * Transforms a document or sentence to its vector-average representation
    * @param  doc token sequence of a document
    * @return vector-average representation of document
    */
  def transform[S <: Iterable[String]](doc: S): Vector = {
    if (doc.isEmpty) {
      Vectors.sparse(vectorSize, Array.empty[Int], Array.empty[Double])
    } else {
      val sum = new Array[Float](vectorSize)
      val sentence = doc.toSeq
      sentence.foreach { word =>
        wordIndex.get(word).foreach { index =>
          val vec = wordVectors.slice(index * vectorSize, (index + 1) * vectorSize)
          blas.saxpy(vectorSize, 1.0f, vec, 0, 1, sum, 0, 1)
        }
      }
      blas.sscal(vectorSize, 1.0f / sentence.size, sum, 0, 1)
      Vectors.dense(sum.map(_.toDouble))
    }
  }
  /**
    * Find synonyms of a word; do not include the word itself in results.
    * @param word a word
    * @param num number of synonyms to find
    * @return array of (word, cosineSimilarity)
    */
  def findSynonyms(word: String, num: Int): Array[(String, Double)] = {
    val vector = transform(word)
    findSynonyms(vector, num, Some(word))
  }
  /**
    * Find synonyms of the vector representation of a word, possibly
    * including any words in the model vocabulary whose vector respresentation
    * is the supplied vector.
    * @param vector vector representation of a word
    * @param num number of synonyms to find
    * @return array of (word, cosineSimilarity)
    */
  def findSynonyms(vector: Vector, num: Int): Array[(String, Double)] = {
    findSynonyms(vector, num, None)
  }
  /**
    * Returns a map of words to their vector representations.
    */
  def getVectors: Map[String, Vector] = {
    wordIndex.map { case (word, ind) =>
      val vec = wordVectors.slice(vectorSize * ind, vectorSize * ind + vectorSize).map(_.toDouble)
      (word, Vectors.dense(vec))
    }
  }
  /**
    * Returns a map of words to their frequency.
    */
  def getVocab: Map[String, Int] = wordCount

  /**
    * Find synonyms of the vector representation of a word, rejecting
    * words identical to the value of wordOpt, if one is supplied.
    * @param vector vector representation of a word
    * @param num number of synonyms to find
    * @param wordOpt optionally, a word to reject from the results list
    * @return array of (word, cosineSimilarity)
    */
  private def findSynonyms(vector: Vector, num: Int, wordOpt: Option[String]): Array[(String, Double)] = {
    require(num > 0, "Number of similar words should > 0")

    val fVector = vector.toArray.map(_.toFloat)
    val cosineVec = new Array[Float](numWords)
    val alpha: Float = 1
    val beta: Float = 0
    // Normalize input vector before blas.sgemv to avoid Inf value
    val vecNorm = blas.snrm2(vectorSize, fVector, 1)
    if (vecNorm != 0.0f) {
      blas.sscal(vectorSize, 1 / vecNorm, fVector, 0, 1)
    }
    blas.sgemv(
      "T", vectorSize, numWords, alpha, wordVectors, vectorSize, fVector, 1, beta, cosineVec, 1)

    var i = 0
    while (i < numWords) {
      val norm = wordVecNorms(i)
      if (norm == 0.0f) {
        cosineVec(i) = 0.0f
      } else {
        cosineVec(i) /= norm
      }
      i += 1
    }

    val pq = new BoundedPriorityQueue[(String, Float)](num + 1)(Ordering.by(_._2))

    var j = 0
    while (j < numWords) {
      pq += Tuple2(wordList(j), cosineVec(j))
      j += 1
    }

    val scored = pq.toSeq.sortBy(-_._2)

    val filtered = wordOpt match {
      case Some(w) => scored.filter(tup => w != tup._1)
      case None => scored
    }

    filtered
      .take(num)
      .map { case (word, score) => (word, score.toDouble) }
      .toArray
  }

  override protected def formatVersion = "1.0"

  def save(sc: SparkContext, path: String): Unit = {
    val vec = getVectors.mapValues(_.toArray.map(_.toFloat))
    Word2VecModel.SaveLoadV1_0.save(sc, path, vec, wordCount)
  }
}

object Word2VecModel extends Loader[Word2VecModel] {

  private def buildWordIndex(model: Map[String, Array[Float]]): Map[String, Int] = {
    model.keys.zipWithIndex.toMap
  }

  private def buildWordVectors(model: Map[String, Array[Float]]): Array[Float] = {
    require(model.nonEmpty, "Word2VecMap should be non-empty")
    val (vectorSize, numWords) = (model.head._2.length, model.size)
    val wordList = model.keys.toArray
    val wordVectors = new Array[Float](vectorSize * numWords)
    var i = 0
    while (i < numWords) {
      Array.copy(model(wordList(i)), 0, wordVectors, i * vectorSize, vectorSize)
      i += 1
    }
    wordVectors
  }

  private object SaveLoadV1_0 {

    val formatVersionV1_0 = "1.0"

    val classNameV1_0 = "org.apache.spark.mllib.feature.nlp.Word2VecModel"

    case class Data(word: String, vector: Array[Float], count: Int)

    def load(sc: SparkContext, path: String): Word2VecModel = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val dataFrame = spark.read.parquet(Loader.dataPath(path))
      // Check schema explicitly since erasure makes it hard to use match-case for checking.
      Loader.checkSchema[Data](dataFrame.schema)

      val dataArray = dataFrame.select("word", "vector").collect()
      val vocabArray = dataFrame.select("word","count").collect()
      val word2VecMap = dataArray.map(i => (i.getString(0), i.getSeq[Float](1).toArray)).toMap
      val wordCountMap = vocabArray.map(i => (i.getString(0), i.getInt(1))).toMap

      new Word2VecModel(word2VecMap, wordCountMap)
    }

    def save(sc: SparkContext, path: String, model: Map[String, Array[Float]],
             wordCount: Map[String, Int]): Unit = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      import spark.implicits._

      val vectorSize = model.values.head.length
      val numWords = model.size
      val metadata = compact(render(
        ("class" -> classNameV1_0) ~ ("version" -> formatVersionV1_0) ~
          ("vectorSize" -> vectorSize) ~ ("numWords" -> numWords)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))

      // We want to partition the model in partitions smaller than
      // spark.kryoserializer.buffer.max
      val bufferSize = Utils.byteStringAsBytes(
        spark.conf.get("spark.kryoserializer.buffer.max", "64m"))
      // We calculate the approximate size of the model
      // We only calculate the array size, considering an
      // average string size of 15 bytes, the formula is:
      // (floatSize * vectorSize + 15) * numWords
      val approxSize = (4L * vectorSize + 15) * numWords
      val nPartitions = ((approxSize / bufferSize) + 1).toInt

      sc.parallelize(model.toSeq).join(sc.parallelize(wordCount.toSeq))
        .map{case(word,(vec, count)) => (word, vec, count)}
        .repartition(nPartitions)
        .toDF("word", "vector", "count")
        .write.parquet(Loader.dataPath(path))
    }
  }

  override def load(sc: SparkContext, path: String): Word2VecModel = {

    val (loadedClassName, loadedVersion, metadata) = Loader.loadMetadata(sc, path)

    implicit val formats = DefaultFormats
    val expectedVectorSize = (metadata \ "vectorSize").extract[Int]
    val expectedNumWords = (metadata \ "numWords").extract[Int]
    val classNameV1_0 = SaveLoadV1_0.classNameV1_0
    (loadedClassName, loadedVersion) match {
      case (classNameV1_0 , "1.0") =>
        val model = SaveLoadV1_0.load(sc, path)
        val vec = model.getVectors
          .mapValues(v => v.toArray.map(_.toFloat))
        val vectorSize = vec.values.head.length
        val numWords = vec.size
        require(expectedVectorSize == vectorSize,
          s"Word2VecModelNew requires each word to be mapped to a vector of size " +
            s"$expectedVectorSize, got vector of size $vectorSize")
        require(expectedNumWords == numWords,
          s"Word2VecModelNew requires $expectedNumWords words, but got $numWords")
        model
      case _ => throw new Exception(
        s"Word2VecModelNew.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $loadedVersion).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)")
    }
  }
}


