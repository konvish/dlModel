package scala

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.{Log, Sigmoid}
import org.nd4j.linalg.factory.Nd4j

class LogisticRegulations(var weight: INDArray, var bias: INDArray) {
  val learningRate = 0.01

  /**
    * 批量梯度下降求解
    *
    * @param trainData 数据
    * @param label     label
    * @return
    */
  def fitBGD(trainData: INDArray, label: INDArray): Double = {
    val z = trainData.mmul(weight.transpose()).add(bias)
    val a = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z.dup()))
    val dz = a.dup().sub(label)
    weight.subi(trainData.mulColumnVector(dz).mul(learningRate).sum(0))
    bias.subi(dz.mul(learningRate).sum(0))
    val left = label.mul(Nd4j.getExecutioner.execAndReturn(new Log(a)))
    val right = label.mul(-1).add(1).mul(Nd4j.getExecutioner.execAndReturn(new Log(a.mul(-1).add(1))))
    val diff = left.add(right).mul(-1)
    diff.sumNumber().doubleValue() / label.length()
  }

  /**
    * 批量梯度下降求解
    *
    * @param trainData 数据
    * @param label     label
    * @return
    */
  def fitSGD(trainData: INDArray, label: INDArray): Double = {
    var diff = 0.0
    for (i <- 0 until label.length()) {
      val z = trainData.getRow(i).mmul(weight.transpose()).add(bias)
      val a = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z.dup()))
      val dz = a.dup().sub(label.getRow(i))
      val v = trainData.getRow(i).dup().mul(dz).mul(learningRate)
      val vv = dz.mul(learningRate)
      weight.subi(v)
      bias.subi(vv)
      diff += -(label.getRow(i).getDouble(0) * math.log(a.getDouble(0)) + (1 - label.getRow(i).getDouble(0)) * math.log(1 - a.getDouble(0)))
    }
    diff / label.length()
  }
}

object LogisticRegulations {

  def main(args: Array[String]): Unit = {
    val train = Nd4j.create(Array(0.10, 1.0, 0.10, -0.20, -0.6, -0.3, 0.2, 1.0, 0.1, 0.2, -0.6, -0.3,
      -2.0, -6.0, 0.0, -2.0, 0.6, -0.3, 0.1, 1.0, 0.1, -0.2, -0.6, -0.3, 0.1, 1.5, 0.1, 0.1, 1.0, 0.5), Array(10, 3))
    val weight = Nd4j.create(Array(-100.0, 200.0, 700.0), Array(1, 3))
    val bias = Nd4j.create(Array(1.0), Array(1, 1))
    val label = Nd4j.create(Array(0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0), Array(10, 1))
    val lr = new LogisticRegulations(weight, bias)
    for (i <- 0 to 10000) {
      val model = lr.fitSGD(train, label)
      println(model)
      println(lr.weight)
      println(lr.bias)
    }

    //bgd
    val weight1 = Nd4j.create(Array(-150.36,  -204.04,  572.19), Array(1, 3))
    val bias1 = Nd4j.create(Array(60.74), Array(1, 1))
    val z1 = train.mmul(weight1.transpose()).add(bias1)
    val a1 = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z1.dup()))
    println(a1)

    //sgd
    val weight2 = Nd4j.create(Array(-150.45,  -203.99,  572.44), Array(1, 3))
    val bias2 = Nd4j.create(Array(60.77), Array(1, 1))
    val z2 = train.mmul(weight2.transpose()).add(bias2)
    val a2 = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z2.dup()))
    println(a2)
  }
}
