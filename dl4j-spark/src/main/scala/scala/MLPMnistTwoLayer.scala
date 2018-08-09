package scala

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory

object MLPMnistTwoLayer {
  private val log = LoggerFactory.getLogger(MLPMnistTwoLayer.getClass)

  def main(args: Array[String]): Unit = {
    val numRows = 28
    val numColumns = 28
    val outputNum = 10
    val rngSeed = 123
    val rate = 0.0015
    val batchSize = 64
    val numEpochs = 15

    val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)

    log.info("Build model....")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .activation(Activation.RELU)
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.98)).l2(rate * 0.005)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(numRows * numColumns)
        .nOut(500)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(500)
        .nOut(100)
        .build())
      .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .nIn(100)
        .nOut(outputNum)
        .build())
      .pretrain(false).backprop(true)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(5))

    log.info("Train model....")
    for (i <- 0 until numEpochs) {
      log.info("Epoch " + i)
      model.fit(mnistTrain)
    }

    log.info("Evaluate model....")
    val eval = new Evaluation(outputNum)
    while (mnistTest.hasNext) {
      val next = mnistTest.next()
      val output = model.output(next.getFeatureMatrix)
      eval.eval(next.getLabels, output)
    }

    log.info(eval.stats())
    log.info("***********************finished***********************")
  }
}
