package cn.sibat.dl.scala

import java.util.Random

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{BackpropType, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

object ComGraphLSTMExample {

  def sampleCharactersFromNetwork(generationInitialization: String, net: ComputationGraph, iter: CharacterIterator, rng: Random, nCharactersToSample: Int, nSamplesToGenerate: Int): Array[String] = {
    var initialization = generationInitialization
    if (generationInitialization == null) {
      initialization = String.valueOf(iter.getRandomCharacter)
    }

    val initializationInput = Nd4j.zeros(nSamplesToGenerate, iter.inputColumns(), initialization.length)
    val init = initialization.toCharArray
    for (i <- 0 until init.length) {
      val idx = iter.convertCharToIndex(init(i))
      for (j <- 0 until nSamplesToGenerate) {
        initializationInput.putScalar(Array(j, idx, i), 1.0)
      }
    }

    val sbs = new Array[StringBuilder](nSamplesToGenerate)
    for (i <- 0 until nSamplesToGenerate) sbs(i) = new StringBuilder(initialization)

    net.rnnClearPreviousState()
    var output = net.rnnTimeStep(initializationInput)(0)
    output = output.tensorAlongDimension(output.size(2) - 1, 1, 0)

    for (i <- 0 until nCharactersToSample) {
      val nextInput = Nd4j.zeros(nSamplesToGenerate, iter.inputColumns())
      for (j <- 0 until nSamplesToGenerate) {
        val outputProbDistribution = new Array[Double](iter.totalOutcomes())
        for (k <- outputProbDistribution.indices)
          outputProbDistribution(k) = output.getDouble(j, k)
        val sampledCharacterIdx = GravesLSTMCharModelling.sampleFromDistribution(outputProbDistribution, rng)
        nextInput.putScalar(Array(j, sampledCharacterIdx), 1.0)
        sbs(j).append(iter.convertIndexToCharacter(sampledCharacterIdx))
      }

      output = net.rnnTimeStep(nextInput)(0)
    }

    val out = new Array[String](nSamplesToGenerate)
    for (i <- 0 until nSamplesToGenerate)
      out(i) = sbs(i).toString()
    out
  }

  def main(args: Array[String]): Unit = {
    val lstmLayerSize = 200
    val miniBatchSize = 32
    val exampleLength = 1000
    val tbpttLength = 50
    val numEpochs = 1
    val generateSamplesEveryNMiniBatches = 10
    val nSamplesToGenerate = 4
    val nCharactersToSample = 4
    val generationInitialization: String = null
    val rng = new Random(12345)

    val iter = GravesLSTMCharModelling.getShakespeareIterator(miniBatchSize, exampleLength)
    val nOut = iter.totalOutcomes()

    val conf = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.1)
      .seed(12345)
      .regularization(true)
      .l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .graphBuilder()
      .addInputs("input")
      .addLayer("first", new GravesLSTM.Builder()
        .nIn(iter.inputColumns())
        .nOut(lstmLayerSize)
        .updater(Updater.RMSPROP)
        .activation(Activation.TANH)
        .build(), "input")
      .addLayer("second", new GravesLSTM.Builder()
        .nIn(lstmLayerSize)
        .nOut(lstmLayerSize)
        .updater(Updater.RMSPROP)
        .activation(Activation.TANH)
        .build(), "first")
      .addLayer("outputLayer", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .updater(Updater.RMSPROP)
        .nIn(2 * lstmLayerSize)
        .nOut(nOut)
        .build(), "first", "second")
      .setOutputs("outputLayer")
      .backpropType(BackpropType.TruncatedBPTT)
      .tBPTTForwardLength(tbpttLength)
      .tBPTTBackwardLength(tbpttLength)
      .pretrain(false)
      .backprop(true)
      .build()

    val net = new ComputationGraph(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    var totalNumParams = 0
    for (i <- 0 until net.getNumLayers) {
      val nParams = net.getLayer(i).numParams()
      println("Number of parameters in layer " + i + ": " + nParams)
      totalNumParams += nParams
    }
    println("Total number of network parameters: " + totalNumParams)

    var miniBatchNumber = 0
    for (i <- 0 until numEpochs) {
      while (iter.hasNext) {
        val ds = iter.next()
        net.fit(ds)
        miniBatchNumber += 1
        if (miniBatchNumber % generateSamplesEveryNMiniBatches == 0) {
          println("-------------------------------------")
          println("Completed " + miniBatchNumber + " miniBatches of size " + miniBatchSize + "x" + exampleLength + " characters")
          println("Sampling characters from network given initialization \"" + generationInitialization + "\"")
          val samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate)
          for (j <- samples.indices) {
            println("---------Sample " + j + "-----------")
            println(samples(j))
            println()
          }
        }
      }

      iter.reset()
    }

    println("\n\nExample complete")
  }
}
