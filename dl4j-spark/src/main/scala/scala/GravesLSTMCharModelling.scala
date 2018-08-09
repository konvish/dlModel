package scala

import java.io.{File, IOException}
import java.net.URL
import java.nio.charset.Charset

import org.apache.commons.io.FileUtils
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{BackpropType, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.util.Random

object GravesLSTMCharModelling {

  def getShakespeareIterator(miniBatchSize: Int, exampleLength: Int): CharacterIterator = {
    val url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt"
    val tempDir = System.getProperty("java.io.tmpdir")
    val fileLocation = tempDir + "Shakespeare.txt"
    val f = new File(fileLocation)
    if (!f.exists()) {
      FileUtils.copyURLToFile(new URL(url), f)
      println("File downloaded to " + f.getAbsolutePath)
    } else {
      println("Using existing text file at " + f.getAbsolutePath)
    }

    if (!f.exists()) throw new IOException("File does not exist: " + fileLocation)

    val validCharacters = CharacterIterator.getMinimalCharacterSet
    new CharacterIterator(fileLocation, Charset.forName("UTF-8"), miniBatchSize, exampleLength, validCharacters, new java.util.Random(12345))
  }

  def sampleFromDistribution(outputProbDistribution: Array[Double], rng: Random): Int = {
    var d = 0.0
    var sum = 0.0
    for (i <- 0 until 10) {
      d = rng.nextDouble()
      sum = 0.0
      for (j <- outputProbDistribution.indices) {
        sum += outputProbDistribution(j)
        if (d <= sum) return i
      }
    }
    throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum)
  }

  def sampleCharactersFromNetwork(generationInitialization: String, model: MultiLayerNetwork, iter: CharacterIterator, rng: Random, nCharactersToSample: Int, nSamplesToGenerate: Int): Array[String] = {
    var initialization = generationInitialization
    if (initialization == null) {
      initialization = String.valueOf(iter.getRandomCharacter)
    }

    val initializationInput = Nd4j.zeros(nSamplesToGenerate, iter.inputColumns(), initialization.length)
    val init = initialization.toCharArray
    for (i <- init.indices) {
      val idx = iter.convertCharToIndex(init(i))
      for (j <- 0 until nSamplesToGenerate) {
        initializationInput.putScalar(Array(j, idx, i), 1.0)
      }
    }

    val sbs = new Array[StringBuilder](nSamplesToGenerate)
    for (i <- 0 until nSamplesToGenerate) sbs(i) = new StringBuilder(initialization)

    model.rnnClearPreviousState()
    var output = model.rnnTimeStep(initializationInput)
    output = output.tensorAlongDimension(output.size(2) - 1, 1, 0)

    for (i <- 0 until nCharactersToSample) {
      val nextInput = Nd4j.zeros(nSamplesToGenerate, iter.inputColumns())
      for (j <- 0 until nSamplesToGenerate) {
        val outputProbDistribution = new Array[Double](iter.totalOutcomes())
        for (k <- outputProbDistribution.indices) {
          outputProbDistribution(k) = output.getDouble(j, k)
        }
        val sampleCharacterIdx = sampleFromDistribution(outputProbDistribution, rng)

        nextInput.putScalar(Array(j, sampleCharacterIdx), 1.0)
        sbs(j).append(iter.convertIndexToCharacter(sampleCharacterIdx))
      }

      output = model.rnnTimeStep(nextInput)
    }

    val out = new Array[String](nSamplesToGenerate)
    for (i <- out.indices) out(i) = sbs(i).toString()
    out
  }

  def main(args: Array[String]): Unit = {
    val lstmLayerSize = 200
    val miniBatchSize = 32
    val exampleLength = 1000
    val tbpttLength = 50
    val numEpochs = 1
    val generateSamplesEveryMinibatches = 10
    val nSamplesToGenerate = 4
    val nCharactersToSample = 300
    val generationInitialization: String = null

    val rng = new Random(12345)
    val iter = getShakespeareIterator(miniBatchSize, exampleLength)
    val nOut = iter.totalOutcomes()

    val conf = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .seed(12345).l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.RMSPROP)
      .list()
      .layer(0, new GravesLSTM.Builder()
        .nIn(iter.inputColumns())
        .nOut(lstmLayerSize)
        .activation(Activation.TANH)
        .build())
      .layer(1, new GravesLSTM.Builder()
        .nIn(lstmLayerSize)
        .nOut(lstmLayerSize)
        .activation(Activation.TANH)
        .build())
      .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(lstmLayerSize)
        .nOut(nOut)
        .build())
      .backpropType(BackpropType.TruncatedBPTT)
      .tBPTTForwardLength(tbpttLength)
      .tBPTTBackwardLength(tbpttLength)
      .pretrain(false)
      .backprop(true)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(1))

    val layers = model.getLayers
    var totalNumParams = 0
    for (i <- layers.indices) {
      val nParams = layers(i).numParams()
      println("Number of parameters in layer " + i + ": " + nParams)
      totalNumParams += nParams
    }

    println("Total number of network parameters: " + totalNumParams)

    var miniBatchNumber = 0
    for (i <- 0 until numEpochs) {
      while (iter.hasNext) {
        val ds = iter.next()
        model.fit(ds)
        miniBatchNumber += 1
        if (miniBatchNumber % generateSamplesEveryMinibatches == 0) {
          println("------------------------------------------")
          println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters")
          println("Sampling characters from network given initialization \"" + generateSamplesEveryMinibatches + "\"")
          val samples = sampleCharactersFromNetwork(generationInitialization, model, iter, rng, nCharactersToSample, nSamplesToGenerate)
          for (j <- samples.indices) {
            println("--------- Sample " + j + "----------")
            println(samples(j))
            println()
          }
        }
      }
      iter.reset()
    }
    println("\n\nExample compete")
  }
}
