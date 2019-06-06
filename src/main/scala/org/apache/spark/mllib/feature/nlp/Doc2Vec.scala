package org.apache.spark.mllib.feature.nlp

import java.lang.{Iterable => JavaIterable}

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd._
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.util.Utils
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.control.Breaks._


class Doc2Vec() extends Serializable with Logging {

  private var vectorSize = 100
  private var learningRate = 0.025
  private var numPartitions = 1
  private var numIterations = 1
  private var seed = Utils.random.nextLong()
  private var minCount = 5
  private var window = 5

  private var decay = 1d
  private var sample = 1e-3
  private var maxVocabSize = 0
  private var customDic = "__customDic__"

  private var useCustomDic = false
  private var minReduce = 1

  def setWindow(window: Int): this.type = {
    this.window = window
    this
  }
  def setDecay(decay: Double): this.type = {
    this.decay = decay
    this
  }
  def setVectorSize(vectorSize: Int): this.type = {
    this.vectorSize = vectorSize
    this
  }
  def setLearningRate(learningRate: Double): this.type = {
    require(learningRate > 0,
      s"Initial learning rate must be positive but got $learningRate")
    this.learningRate = learningRate
    this
  }
  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0,
      s"Number of partitions must be positive but got $numPartitions")
    this.numPartitions = numPartitions
    this
  }
  def setNumIterations(numIterations: Int): this.type = {
    require(numIterations >= 0,
      s"Number of iterations must be nonnegative but got $numIterations")
    this.numIterations = numIterations
    this
  }
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }
  def setSample(sample: Double): this.type = {
    require(sample>=0, s"sample must be nonegative but got $sample")
    this.sample = sample
    this
  }
  def setMaxVocabSize(size: Int): this.type = {
    require(sample>=0, s"maxVocabSize must be nonegative but got $sample")
    this.maxVocabSize = size
    this
  }
  def setCustomDic(dic: String): this.type = {
    this.customDic = dic
    this
  }
  def setMinCount(minCount: Int): this.type = {
    require(minCount >= 0,
      s"Minimum number of times must be nonnegative but got $minCount")
    this.minCount = minCount
    this
  }

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val MAX_CODE_LENGTH = 40


  private var vocab: Array[VocabWord] = null
  private var vocabHash = mutable.HashMap.empty[String, Int]
  private var vocabSize = 0
  private var trainWordsCount = 0L

  private def learnVocab[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    val sc = dataset.sparkContext
    if(useCustomDic) {
      val dic = sc.textFile(customDic).map(_.split(" "))
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

    for (i <- 0 until EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
    }
    expTable
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

  def fit[S <: Iterable[String]](dataset: RDD[(Int, S)]): Doc2VecModel = {
    if(customDic != "__customDic__") useCustomDic = true
    val sc = dataset.context

    learnVocab(dataset.map(_._2.toSeq))
    createBinaryTree()

    val expTable = sc.broadcast(createExpTable())
    val bcVocab = sc.broadcast(vocab)
    val bcVocabHash = sc.broadcast(vocabHash)
    try {
      doFit(dataset, sc, expTable,bcVocab, bcVocabHash)
    } finally {
      expTable.destroy(blocking = false)
      bcVocab.destroy(blocking = false)
      bcVocabHash.destroy(blocking = false)
    }
  }

  private def doFit[S <: Iterable[String]](
    dataset: RDD[(Int, S)],
    sc: SparkContext,
    expTable: Broadcast[Array[Float]],
    bcVocab: Broadcast[Array[VocabWord]],
    bcVocabHash: Broadcast[mutable.HashMap[String, Int]]): Doc2VecModel = {

    val documents: RDD[(Int, Array[Int])] = dataset.mapPartitions { docIter =>
      docIter.map { case(id, doc) =>
        val wordIndexes = doc.flatMap(bcVocabHash.value.get).toArray
        (id, wordIndexes)
      }
    }.repartition(numPartitions).cache()

    val docSize = documents.count().toInt
    val initRandom = new XORShiftRandom(seed)
    //init the vectors
    val syn0Global = Array.fill[Float](docSize * vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    //init extra vectors
    val syn1Global = new Array[Float](vocabSize * vectorSize)
    //init the word vectors
    val syn2Global = new Array[Float](vocabSize * vectorSize)


    var docAlpha = learningRate
    var wordAlpha = learningRate
    val totalWordsCounts = numIterations * trainWordsCount + 1
    val startTime = System.currentTimeMillis()

    for(k <- 1 to numIterations) {
      docAlpha = decay * docAlpha
      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)
      val bcSyn2Global = sc.broadcast(syn2Global)
      val numWordsProcessedInPreviousIterations = (k - 1) * trainWordsCount

      val partial = documents.mapPartitionsWithIndex { case (idx, iter) =>
        val random = new XORShiftRandom(seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))
        val syn0Modify = new Array[Int](docSize)
        val syn1Modify = new Array[Int](vocabSize)
        val syn2Modify = new Array[Int](vocabSize)
        val neu1e = new Array[Float](vectorSize)

        val model = iter.foldLeft((bcSyn0Global.value, bcSyn1Global.value, bcSyn2Global.value, 0L, 0L)) {
          case ((syn0, syn1, syn2,lastWordCount, wordCount), (id, doc)) =>
            require(id < docSize)
            var lwc = lastWordCount
            var wc = wordCount
            if (wordCount - lastWordCount > 10000) {
              lwc = wordCount
              val wordCountActual = numPartitions * wordCount.toDouble + numWordsProcessedInPreviousIterations
              wordAlpha = learningRate * (1 - wordCountActual / totalWordsCounts)
              if (wordAlpha < learningRate * 0.0001) wordAlpha = learningRate * 0.0001
              val runTime = (System.currentTimeMillis() - startTime + 1).toDouble / 1000
              val pro = wordCountActual / totalWordsCounts * 100
              val progress = if(pro > 100) 100 else pro
              logInfo(s"wordCount = ${wordCountActual.toInt}, " + f"docAlpha = $docAlpha%.5f, " +
                f"wordAlpha = $wordAlpha%.5f, " + f"runtime = $runTime%.2f S, " + f"progress = $progress%.2f%%, ")
            }
            wc += doc.length
            val docIndex = id * vectorSize
            for (pos <- doc.indices) {
              breakable {
                val word = doc(pos)
                if (sample > 0) {
                  val ran = (math.sqrt(bcVocab.value(word).cn / (sample * trainWordsCount)) + 1) *
                    (sample * trainWordsCount) / bcVocab.value(word).cn
                  if (ran < random.nextDouble()) break()
                }
                val b = random.nextInt(window)
                for (i <- b - window to window - b if i != 0) {
                  val c = pos + i
                  if (c >= 0 && c < doc.length) {
                    val lastWord = doc(c)
                    val l1 = lastWord * vectorSize
                    blas.sscal(vectorSize, 0f, neu1e, 1)
                    for (j <- 0 until bcVocab.value(word).codeLen) {
                      val inner = bcVocab.value(word).point(j)
                      val l2 = inner * vectorSize
                      val f = blas.sdot(vectorSize, syn2, l1, 1, syn1, l2, 1)
                      val sig = if (f > MAX_EXP) 1 else if (f < -MAX_EXP) 0 else {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                        expTable.value(ind)
                      }
                      val g = ((1 - bcVocab.value(word).code(j) - sig) * wordAlpha).toFloat
                      blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                      blas.saxpy(vectorSize, g, syn2, l1, 1, syn1, l2, 1)
                      syn1Modify(inner) += 1
                    }
                    blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn2, l1, 1)
                    syn2Modify(lastWord) += 1
                  }
                }
                blas.sscal(vectorSize, 0f, neu1e,1)
                for (i <- 0 until bcVocab.value(word).codeLen) {
                  val inner = bcVocab.value(word).point(i)
                  val l2 = inner * vectorSize
                  val f = blas.sdot(vectorSize, syn0, docIndex, 1, syn1, l2, 1)
                  val sig = if (f > MAX_EXP) 1 else if (f < -MAX_EXP) 0 else {
                    val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                    expTable.value(ind)
                  }
                  val g = ((1 - bcVocab.value(word).code(i) - sig) * docAlpha).toFloat
                  blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                  blas.saxpy(vectorSize, g, syn0, docIndex, 1, syn1, l2, 1)
                  syn1Modify(inner) += 1
                }
                blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, docIndex, 1)
              }
            }
            syn0Modify(id) += 1
            (syn0, syn1, syn2, lwc, wc)
        }

        val syn0Local = model._1
        val syn1Local = model._2
        val syn2Local = model._3
        // Only output modified vectors.
        Iterator.tabulate(docSize) { index =>
          if (syn0Modify(index) > 0) {
            Some((index,
              (syn0Local.slice(index * vectorSize, (index + 1) * vectorSize), 1)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (syn1Modify(index) > 0) {
            Some((index + docSize,
              (syn1Local.slice(index * vectorSize, (index + 1) * vectorSize), 1)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (numIterations > 1 && syn2Modify(index) > 0) {
            Some((index + docSize + vocabSize,
              (syn2Local.slice(index * vectorSize, (index + 1) * vectorSize), 1)))
          } else {
            None
          }
        }.flatten
      }

      val synAgg = partial.reduceByKey { case ((v1, n1), (v2, n2)) =>
        blas.saxpy(vectorSize, 1.0f, v2, 1, v1, 1)
        (v1, n1 + n2)
      }.mapValues { case(v, n) =>
        if(n > 1) blas.sscal(vectorSize, 1 / n.toFloat, v, 1)
        v
      }.collect()

      for (word <- synAgg) {
        val index = word._1
        val vec = word._2
        if (index < docSize) {
          Array.copy(vec, 0, syn0Global, index * vectorSize, vectorSize)
        } else if (index < docSize + vocabSize){
          Array.copy(vec, 0, syn1Global, (index - docSize) * vectorSize, vectorSize)
        } else {
          Array.copy(vec, 0, syn2Global, (index - docSize - vocabSize) * vectorSize, vectorSize)
        }
      }

      bcSyn0Global.destroy(false)
      bcSyn1Global.destroy(false)
      bcSyn2Global.destroy(false)
    }

    documents.unpersist()
    new Doc2VecModel(syn0Global, syn1Global, docSize, vectorSize, vocab, vocabHash,
      vocabSize, learningRate, numIterations, decay)
  }
}


class Doc2VecModel private[spark](private[spark] val syn0: Array[Float],
                                  private[spark] val syn1: Array[Float],
                                  private[spark] val docSize: Int,
                                  private[spark] val vectorSize: Int,
                                  private[spark] val vocab: Array[VocabWord],
                                  private[spark] val vocabHash: mutable.HashMap[String, Int],
                                  private[spark] val vocabSize: Int,
                                  private[spark] val learningRate: Double,
                                  private[spark] val numIterations: Int,
                                  private[spark] val decay: Double) extends Serializable {

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6

  private def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](EXP_TABLE_SIZE)
    for (i <- 0 until EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
    }
    expTable
  }

  def transform(docId: Int): Vector = {
    require(docId < docSize)
    Vectors.dense(syn0.slice(docId * vectorSize, (docId + 1) * vectorSize).map(_.toDouble))
  }
  def transform[S <: Iterable[String]](dataset: S): Vector = {
    val doc: Array[Int] = dataset.flatMap(vocabHash.get).toArray

    val initRandom = new XORShiftRandom()
    val expTable = createExpTable()
    val vector = Array.fill[Float](vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)

    var alpha = learningRate

    for(_ <- 0 until numIterations) {
      val neu1e = new Array[Float](vectorSize)
      for (pos <- doc.indices) {
        val word = doc(pos)
        blas.sscal(vectorSize, 0f, neu1e,1)
        for (i <- 0 until vocab(word).codeLen) {
          val inner = vocab(word).point(i)
          val l2 = inner * vectorSize
          val f = blas.sdot(vectorSize, vector, 0, 1, syn1, l2, 1)
          val sig = if (f > MAX_EXP) 1 else if (f < -MAX_EXP) 0 else {
            val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
            expTable(ind)
          }
          val g = ((1 - vocab(word).code(i) - sig) * alpha).toFloat
          blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
        }
        blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, vector, 0, 1)
      }
      alpha = decay * alpha
    }

    Vectors.dense(vector.map(_.toDouble))
  }

  def getVectors: Map[Int, Vector] = {
    val docVectors = mutable.HashMap.empty[Int, Vector]
    for (i <- 0 until docSize) {
      val vec = syn0.slice(i * vectorSize, i * vectorSize + vectorSize)
      docVectors += i -> Vectors.dense(vec.map(_.toDouble))
    }
    docVectors.toMap
  }
}



