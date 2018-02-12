package cn.sibat.test

import org.nd4j.linalg.factory.Nd4j

object Test {
  def main(args: Array[String]): Unit = {
    val weight = Nd4j.ones(4, 4)
    val da = Nd4j.create(Array(0.1, 0.2, 0.3, 0.4))
    val dz_1 = Nd4j.ones(4, 1)
    val dz = weight.mmul(dz_1).mul(da.transpose())
    println(dz)
  }
}
