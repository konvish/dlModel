package cn.sibat.rl4j.scala.game

import org.json.{JSONArray, JSONObject}

import scala.collection.mutable.ArrayBuffer

object RunMain {
  private var J = false
  private var headless = true
  private var evalRun = false
  private var q = new Array[GameR](20)
  private var z = new GameMap(7, 70, 7)
  private var A = new GameMap(7, 70, 7)
  private var n = new GameMap(7, 70, 7) //1 + 2 * lanesSide, patchesAhead + patchesBehind, 0
  private var p = 0
  private var y = 0
  private var t = 1.5
  private var E = 0
  private var I = 0
  private var B = 0
  private var v = 0
  private var w = 0
  private var x = 0

  def reset(): Unit = {
    z = new GameMap(7, 70, 7)
    A = new GameMap(7, 70, 7)
    n = new GameMap(7, 70, 7) //1 + 2 * lanesSide, patchesAhead + patchesBehind, 0
    q = new Array[GameR](20)
    y = 0
    t = 1.5
    x = 0
    w = 0
    v = 0
    B = 0
    I = 0
    E = 0
  }

  def L(): Unit = {
    z.reset()
    for (a <- q.indices) q(a).move(0 != a)
  }

  def doEvalRun(a: Int, b: Int, d: () => Unit, c: Int): Unit = {
    val h = c
    headless = true
    val c_1 = J
    evalRun = true
    J = true
    var i_f = 0
    val i_g = new ArrayBuffer[Double]()
    for (i <- 0 until a) {
      reset()
      var j_g = 0.0
      for (j <- 0 until b) {
        if (0 == i_f % h) d()
        //L()
        j_g += q(0).c + q(0).a
        i_f += 1
      }
      i_g += math.floor(j_g / b * 2E3) / 100
    }
    reset()
    J = c_1
    evalRun = false
    headless = false
    val sort = i_g.sorted
    i_g(a / 2)
  }

  def main(args: Array[String]): Unit = {
    val lanesSide = 3
    val patchesAhead = 50
    val patchesBehind = 10
    val trainIterations = 10000
    val num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind)
    val num_actions = 5
    val temporal_window = 0
    val network_szie = num_inputs * temporal_window + num_actions * temporal_window + num_inputs

    val layer_defs = new JSONArray()
    val inputLayer = new JSONObject().put("type", "input").put("out_sx", 1).put("out_sy", 1).put("out_depth", network_szie)
    layer_defs.put(inputLayer)
    (0 to 3).foreach(i => {
      layer_defs.put(new JSONObject().put("type", "fc").put("num_neurons", 24).put("activation", "tanh"))
    })
    layer_defs.put(new JSONObject().put("type", "regression").put("num_neurons", num_actions))

    val tdtrainer_options = new JSONObject().put("learning_rate", 0.001).put("momentum", 0.0).put("batch_size", 128).put("l2_decay", 0.01)

    val opt = new JSONObject()
    opt.put("temporal_window", temporal_window)
    opt.put("experience_size", 10000)
    opt.put("start_learn_threshold", 5000)
    opt.put("gamma", 0.98)
    opt.put("learning_steps_total", 100000)
    opt.put("learning_steps_burnin", 1000)
    opt.put("epsilon_min", 0.0)
    opt.put("epsilon_test_time", 0.0)
    opt.put("layer_defs", layer_defs)
    opt.put("tdtrainer_options", tdtrainer_options)

    val brain = new DeepQLearnBrain(num_actions, num_actions, opt)
    val learn = (state: Array[Double], lastReward: Int) => {
      brain.backward(lastReward)
      val action = brain.forward(state)
      //draw_net()
      //draw_stats()
      action
    }

    if (trainIterations > 0) {
      val totalFrames = 30 * trainIterations
      val numRuns = totalFrames / 100000 + 1
      val percent = 0

    }
    brain.learning = false

  }

  class GameMap(a: Int, b: Int, d: Int) {
    private var data: Array[Array[Int]] = _
    private val defaultValue = d

    def init(): Unit = {
      data = Array.ofDim(a, b)
      for (c <- 0 until a; g <- 0 until b) {
        data(c)(g) = d
      }
    }

    def reset(): Unit = {
      for (a <- data.indices; b <- data(a).indices) {
        data(a)(b) = defaultValue
      }
    }

    def set(a: Int, b: Int, c: Int): Unit = {
      0 <= a && a < this.data.length && 0 <= b && b < this.data(a).length && (this.data(a)(b) == c)
    }

    def get(a: Int, b: Int, c: Int): Int = {
      if (0 <= a && a < this.data.length && 0 <= b && b < this.data(a).length) this.data(a)(b)
      else c
    }

    def m(): Unit = {
      val a = n

    }
  }

  class GameR() {
    var a = 1.0
    var c = 1
    var b = 0
    var y = 10 * math.floor(700 * math.random / 10)
    var x = 0
    var f = new Array[Int](60)
    this.init()

    def init(): Unit = {
      val a = (140 * math.random / 20).toInt
      val b = 1 + 0.7 * math.random
      this.x = 20 * a + 4
      this.a = b
      this.b = a
    }

    def move(num: Boolean): Unit = {
      val b = y - c * a - t
      if (num && 525 > y && 525 <= b) {
        v += 1
        w += 1
        x += 1
      } else if (num && 525 < this.y && 525 >= b) {
        v -= 1
        w -= 1
        x -= 1
      }
    }
  }

}
