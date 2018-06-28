package scala

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.{Log, Sigmoid}
import org.nd4j.linalg.factory.Nd4j

/**
  * 构建深度网络
  *
  * @param n     层数
  * @param array 每一层对应的节点数
  */
class DeepLayerDemo(n: Int, array: Array[Int]) {
  var learningRate = 0.01
  val weight = new Array[INDArray](n)
  val bias = new Array[INDArray](n)

  def setLearningRate(learningRate: Double): Unit = {
    this.learningRate = learningRate
  }

  /**
    * 初始化网络参数
    *
    * @param inputDimension 输入的维度
    */
  def init(inputDimension: Int): Unit = {
    require(n == array.length)
    for (i <- 0 until n) {
      if (i == 0) {
        weight(i) = Nd4j.rand(inputDimension, array(i))
        //weight(i) = Nd4j.ones(inputDimension, array(i))
      } else {
        weight(i) = Nd4j.rand(array(i - 1), array(i))
        //weight(i) = Nd4j.ones(array(i - 1), array(i))
      }
      bias(i) = Nd4j.ones(array(i))
    }
    //    weight(0) = Nd4j.create(Array(0.2, 0.5, 0.7, 0.2, 0.5, 0.7, 0.2, 0.5, 0.7), Array(3, 3))
    //    weight(1) = Nd4j.create(Array(0.2, 0.5, 0.7), Array(3, 1))
  }

  /**
    * 批量梯度下降求解
    *
    * @param trainData 数据
    * @param label     label
    * @return
    */
  def fitBGD(trainData: INDArray, label: INDArray): Double = {
    val z = new Array[INDArray](n)
    val a = new Array[INDArray](n + 1)
    val da = new Array[INDArray](n) //行向量
    val dz = new Array[INDArray](n) //列向量
    val dw = new Array[INDArray](n)
    val db = new Array[INDArray](n)
    a(0) = trainData
    for (i <- 0 until n) {
      z(i) = a(i).mmul(weight(i)).addRowVector(bias(i)) //10*1
      a(i + 1) = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z(i).dup()))
      da(i) = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z(i).dup()).derivative())
    }

    for (i <- n - 1 to 0 by -1) {
      if (i == n - 1)
        dz(i) = a(i + 1).sub(label)
      else {
        dz(i) = weight(i + 1).mmul(dz(i + 1).transpose()).transpose().mul(da(i))
      }
      dw(i) = dz(i).transpose().mmul(a(i)).div(label.length()) //1*3
      db(i) = dz(i).sum(0).div(label.length()).transpose()
    }
    for (j <- 0 until n) {
      weight(j).subi(dw(j).mul(learningRate).transpose())
      bias(j).subi(db(j).mul(learningRate).transpose())
    }
    val left = label.mul(Nd4j.getExecutioner.execAndReturn(new Log(a(n))))
    val right = label.mul(-1).add(1).mul(Nd4j.getExecutioner.execAndReturn(new Log(a(n).mul(-1).add(1))))
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
      val z = new Array[INDArray](n)
      val a = new Array[INDArray](n + 1)
      val da = new Array[INDArray](n) //行向量
      val dz = new Array[INDArray](n) //列向量
      val dw = new Array[INDArray](n)
      val db = new Array[INDArray](n)
      a(0) = trainData.getRow(i)
      for (j <- 0 until n) {
        z(j) = a(j).mmul(weight(j)).add(bias(j)) //1*3
        a(j + 1) = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z(j).dup())) //1*3
        da(j) = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z(j).dup()).derivative()) //1*3
      }
      for (j <- n - 1 to 0 by -1) {
        if (j == n - 1) {
          dz(j) = a(j + 1).dup().sub(label.getRow(i))
        } else {
          dz(j) = weight(j + 1).mmul(dz(j + 1)).mul(da(j).transpose()) //1*3
        }
        dw(j) = dz(j).mmul(a(j)).transpose() //1*3
        db(j) = dz(j)
      }
      for (j <- 0 until n) {
        weight(j).subi(dw(j).mul(learningRate))
        bias(j).subi(db(j).mul(learningRate).transpose())
      }
      diff += -(label.getRow(i).getDouble(0) * math.log(a(n).getDouble(0)) + (1 - label.getRow(i).getDouble(0)) * math.log(1 - a(n).getDouble(0)))
    }
    diff / label.length()
  }
}

object DeepLayerDemo {

  /**
    * 结果验证
    *
    * @param weight    权重
    * @param bias      偏差
    * @param n         层数
    * @param trainData 训练数据
    * @param labelData 标签数据
    * @return
    */
  def validate(weight: Array[INDArray], bias: Array[INDArray], n: Int, trainData: INDArray, labelData: INDArray): INDArray = {
    val z = new Array[INDArray](n)
    val a = new Array[INDArray](n + 1)
    a(0) = trainData
    for (i <- 0 until n) {
      z(i) = a(i).mmul(weight(i)).addRowVector(bias(i)) //10*1
      a(i + 1) = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z(i).dup()))
    }
    a(n)
  }

  def main(args: Array[String]): Unit = {
    val train = Nd4j.create(Array(0.10, 1.0, 0.10, -0.20, -0.6, -0.3, 0.2, 1.0, 0.1, 0.2, -0.6, -0.3,
      -2.0, -6.0, 0.0, -2.0, 0.6, -0.3, 0.1, 1.0, 0.1, -0.2, -0.6, -0.3, 0.1, 1.5, 0.1, 0.1, 1.0, 0.5), Array(10, 3))
    val label = Nd4j.create(Array(0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0), Array(10, 1))
    val n = 4
    //当中间层数的节点为1时，难以学习到规则
    //val lr = new DeepLayerDemo(n, Array(3, 7, 5, 1, 6, 8, 8, 4, 6, 1))
    val lr = new DeepLayerDemo(n, Array(3, 7, 3, 1))
    lr.init(3)
    lr.setLearningRate(0.01)
    for (i <- 0 to 100000) {
      val model = lr.fitSGD(train, label)
      if (i % 1000 == 0)
        println(i + ":" + model)
    }

    println("weight")
    lr.weight.foreach(id => println(id))
    println("bias")
    lr.bias.foreach(id => println(id))

    //    val weight1 = Nd4j.create(Array(2.13, 1.65, 1.83, 3.54, 2.84, 2.53, 1.88, 1.51, 1.18), Array(3, 3))
    //    val weight2 = Nd4j.create(Array(-6.06, -3.93, -3.23), Array(1, 3))
    //    val bias1 = Nd4j.create(Array(-0.88, -0.72, -0.53), Array(1, 3))
    //    val bias2 = Nd4j.create(Array(6.13), Array(1, 1))
    //    val z_1 = train.mmul(weight1.transpose()).addRowVector(bias1) //10*3
    //    val a_1 = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z_1.dup())) //10*3
    //    val z_2 = a_1.mmul(weight2.transpose()).add(bias2) //10*1
    //    val a_2 = Nd4j.getExecutioner.execAndReturn(new Sigmoid(z_2.dup())) //10*1
    //    println(a_2)

    //validate
    println(validate(lr.weight, lr.bias, n, train, label))
  }
}
