package cn.sibat.rl4j.scala.game

import scala.collection.mutable.ArrayBuffer

object CNNUtil {

  class Window(size: Int, minsize: Int) {
    private var v = new ArrayBuffer[Double]()
    private var sum = 0.0

    def add(x: Double): Unit = {
      v += x
      sum += x
      if (v.length > size) {
        val xold = v.head
        v.remove(0)
        sum -= xold
      }
    }

    def get_average(): Double = {
      if (v.length < minsize)
        -1
      else
        sum / v.length
    }

    def reset(): Unit = {
      v.clear()
      sum = 0
    }
  }

}
