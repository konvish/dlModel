package cn.sibat.test

import org.nd4j.linalg.factory.Nd4j

object Test {
  def main(args: Array[String]): Unit = {
    val weight = Nd4j.ones(4, 2)
    val da = Nd4j.rand(4,2)
    println(weight.mul(da))
  }
}
