package scala

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory

object MLPMnistSingleLayer {
  private val log = LoggerFactory.getLogger(MLPMnistSingleLayer.getClass)

  def main(args: Array[String]): Unit = {
    val numRows = 28 //矩阵的行数
    val numColumns = 28 //矩阵的列数
    val outputNum = 10 // 输出的类别
    val batchSize = 128 // 每一步抓取的样例数量
    val rngSeed = 123 // 随机数
    val numEpochs = 15 // 一个epoch的周期
    //batchSize 和 numEpochs必须根据经验选择，而经验则需要通过实验来积累。
    // 每批次处理的数据越多，训练速度越快；epoch的数量越多，遍历数据集的次数越多，
    // 准确率就越高。需要进行实验才能发现最优的数值

    val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)

    log.info("Build model....")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.006)
      .updater(Updater.NESTEROVS)
      .regularization(true).l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(numRows * numColumns)
        .nOut(1000)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
        .nIn(1000)
        .nOut(outputNum)
        .activation(Activation.SOFTMAX)
        .weightInit(WeightInit.XAVIER)
        .build())
      .pretrain(false)
      .backprop(true)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(1))

    log.info("Train model ....")
    for (i <- 0 until numEpochs) {
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
    log.info("************************finished*******************************")
  }
}
