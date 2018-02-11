package cn.sibat.dl.scala

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.{Log, Sigmoid}
import org.nd4j.linalg.factory.Nd4j

class SingleLayerDemo(var weight_1: INDArray, var weight_2: INDArray, var bias_1: INDArray, var bias_2: INDArray) {
  val learningRate = 0.01

  /**
    * 批量梯度下降求解
    *
    * @param trainData 数据
    * @param label     label
    * @return
    */
  def fitBGD(trainData: INDArray, label: INDArray): Double = {
    val z = trainData.mmul(weight_1.transpose()).add(bias_1)
    val a = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z.dup()))
    val dz = a.dup().sub(label)
    weight_1.subi(trainData.mulColumnVector(dz).mul(learningRate).sum(0))
    bias_1.subi(dz.mul(learningRate).sum(0))
    val left = label.mul(Nd4j.getExecutioner.execAndReturn(new Log(a)))
    val right = label.mul(-1).add(1).mul(Nd4j.getExecutioner.execAndReturn(new Log(a.mul(-1).add(1))))
    val diff = left.add(right).mul(-1)
    diff.sumNumber().doubleValue() / label.length()
  }

  /**
    * 随机梯度下降求解
    *
    * @param trainData 数据
    * @param label     label
    * @return
    */
  def fitSGD(trainData: INDArray, label: INDArray): Double = {
    var diff = 0.0
    for (i <- 0 until label.length()) {
      val z_1 = trainData.getRow(i).mmul(weight_1.transpose()).add(bias_1.transpose()) //1*3
      val a_1 = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z_1.dup())) //1*3
      val dga_1 = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z_1.dup()).derivative()) //1*3
      val z_2 = a_1.mmul(weight_2.transpose()).add(bias_2) //1*1
      val a_2 = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z_2.dup())) //1*1
      val dz_2 = a_2.dup().sub(label.getRow(i)) //1*1
      val dw_2 = a_1.mul(dz_2) //1*3
      val db_2 = dz_2 //1*1
      val dz_1 = weight_2.mul(dz_2).mul(dga_1) //1*3
      val dw_1 = dz_1.transpose().mmul(trainData.getRow(i)) //3*3
      val db_1 = dz_1 //1*3
      weight_1.subi(dw_1.mul(learningRate))
      bias_1.subi(db_1.mul(learningRate).transpose())
      weight_2.subi(dw_2.mul(learningRate))
      bias_2.subi(db_2.mul(learningRate))
      diff += -(label.getRow(i).getDouble(0) * math.log(a_2.getDouble(0)) + (1 - label.getRow(i).getDouble(0)) * math.log(1 - a_2.getDouble(0)))
    }
    diff / label.length()
  }
}

object SingleLayerDemo {
  def main(args: Array[String]): Unit = {
    val train = Nd4j.create(Array(0.10, 1.0, 0.10, -0.20, -0.6, -0.3, 0.2, 1.0, 0.1, 0.2, -0.6, -0.3,
      -2.0, -6.0, 0.0, -2.0, 0.6, -0.3, 0.1, 1.0, 0.1, -0.2, -0.6, -0.3, 0.1, 1.5, 0.1, 0.1, 1.0, 0.5), Array(10, 3))
    val weight_1 = Nd4j.create(Array(0.2, 0.5, 0.7, 0.2, 0.5, 0.7, 0.2, 0.5, 0.7), Array(3, 3))
    val weight_2 = Nd4j.create(Array(0.2, 0.5, 0.7), Array(1, 3))
    val bias_1 = Nd4j.ones(3, 1)
    val bias_2 = Nd4j.ones(1)
    val label = Nd4j.create(Array(0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0), Array(10, 1))
    val lr = new SingleLayerDemo(weight_1, weight_2, bias_1, bias_2)
    for (i <- 0 to 10000) {
      val model = lr.fitSGD(train, label)
      println(model)
      println(lr.weight_1)
      println(lr.bias_1)
      println(lr.weight_2)
      println(lr.bias_2)
    }

    //bgd
    val weight1 = Nd4j.create(Array(2.08,  3.46,  2.02, 1.82,  2.98,  1.84, 1.35,  2.09,  1.48), Array(3, 3))
    val weight2 = Nd4j.create(Array(-6.09,  -4.61,  -2.48), Array(1, 3))
    val bias1 = Nd4j.create(Array(-0.86,  -0.73,  -0.39), Array(1, 3))
    val bias2 = Nd4j.create(Array(6.12), Array(1, 1))
    val z_1 = train.mmul(weight1.transpose()).addRowVector(bias1) //10*3
    val a_1 = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z_1.dup())) //10*3
    val z_2 = a_1.mmul(weight2.transpose()).add(bias2) //10*1
    val a_2 = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z_2.dup())) //10*1
    println(a_2)

    //sgd
  }
}
