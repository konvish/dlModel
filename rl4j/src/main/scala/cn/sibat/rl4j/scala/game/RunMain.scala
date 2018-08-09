package cn.sibat.rl4j.scala.game

import org.json.{JSONArray, JSONObject}

import scala.collection.mutable.ArrayBuffer

object RunMain {
  private var J_0 = false
  private var headless_0 = true
  private var evalRun_0 = false
  private var q_0 = new Array[GameAgent](20)
  private var z_0 = new GameMap(7, 70, 7)
  private var A_0 = new GameMap(7, 70, 7)
  private var p_0 = 0
  private var y_0 = 0
  private var t_0 = 1.5
  private var E_0 = 0
  private var I_0 = 0.0
  private var B_0 = 0
  private var v_0 = 0
  private var w_0 = 0
  private var x_0 = 0
  private var m_0 = 0
  private var f_0 = false
  private var k_0 = false
  private var e_0 = false
  private var l_0 = Array(0, 1, 2, 3, 4)
  private val lanesSide = 3
  private val patchesAhead = 50
  private val patchesBehind = 10
  private val trainIterations = 10000
  private val num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind)
  private val num_actions = 5
  private val temporal_window = 0
  private val network_szie = num_inputs * temporal_window + num_actions * temporal_window + num_inputs
  private var n_0 = new GameMap(1 + 2 * lanesSide, patchesAhead + patchesBehind, 0)
  private var brain: DeepQLearnBrain = _

  def initnet(): Unit = {
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

    brain = new DeepQLearnBrain(num_actions, num_actions, opt)
  }

  def learn(state: Array[Double], lastReward: Int): Int = {
    brain.backward(lastReward)
    val action = brain.forward(state)
    //draw_net()
    //draw_stats()
    action
  }

  def reset(): Unit = {
    z_0 = new GameMap(7, 70, 7)
    A_0 = new GameMap(7, 70, 7)
    n_0 = new GameMap(1 + 2 * lanesSide, patchesAhead + patchesBehind, 0)
    q_0 = new Array[GameAgent](20)
    y_0 = 0
    t_0 = 1.5
    x_0 = 0
    w_0 = 0
    v_0 = 0
    B_0 = 0
    I_0 = 0
    E_0 = 0
  }

  def L(): Unit = {
    z_0.reset()
    for (a <- q_0.indices) {
      q_0(a).move(0 != a)
      q_0(a).i(0)
    }
    t_0 = 1.5 - (q_0(0).y_1 - 525) / 1
    for (a <- q_0.indices) {
      q_0(a).l()
      if (a != 0 && math.random > 0.99 + 0.004 * q_0(a).c_1)
        q_0(a).g(if (0.5 < math.random) -1 else 1)
    }
    q_0(0).i(0)
    if (f_0) {
      A_0.reset()
      q_0(0).o()
    }
    I_0 += q_0(0).c_1 * q_0(0).a_1
    if (y_0 % 30 == 0) {
      z_0.updateGameMap(n_0, 1, 1, 1, 1)
      val a = learn(n_0.flatMap(), (I_0.toInt - 60) / 20)
      x_0 = 0
      m_0 = 0
      if (a >= 0 && a < l_0.length) {
        a
      } else {
        B_0 = 0
        I_0 = 0
      }
    }
    q_0(0).j()
    y_0 += 1
  }

  def doEvalRun(a: Int, b: Int, d: () => Unit, c: Int): Unit = {
    val h = c
    headless_0 = true
    val c_1 = J_0
    evalRun_0 = true
    J_0 = true
    var i_f = 0
    val i_g = new ArrayBuffer[Double]()
    for (i <- 0 until a) {
      reset()
      var j_g = 0.0
      for (j <- 0 until b) {
        if (0 == i_f % h) d()
        //L()
        j_g += q_0(0).c_1 + q_0(0).a_1
        i_f += 1
      }
      i_g += math.floor(j_g / b * 2E3) / 100
    }
    reset()
    J_0 = c_1
    evalRun_0 = false
    headless_0 = false
    val sort = i_g.sorted
    i_g(a / 2)
  }

  def main(args: Array[String]): Unit = {
    if (trainIterations > 0) {
      val totalFrames = 30 * trainIterations
      val numRuns = totalFrames / 100000 + 1
      var percent = 0
      doEvalRun(numRuns, totalFrames / numRuns, () => {
        percent += 1
      }, totalFrames / 100)
    }
    brain.learning = false
  }

  class GameMap(length: Int, width: Int, value: Double) {
    private val defaultValue = value
    private val data: Array[Array[Double]] = Array.ofDim(length, width)

    def init(): Unit = {
      for (a <- data.indices; b <- data(a).indices) {
        data(a)(b) = defaultValue
      }
    }

    /**
      * 恢复背景默认状态
      */
    def reset(): Unit = {
      for (a <- data.indices; b <- data(a).indices) {
        data(a)(b) = defaultValue
      }
    }

    /**
      * 把坐标为(i,j)的值设为value
      *
      * @param i     ith
      * @param j     jth
      * @param value 值v
      */
    def set(i: Int, j: Int, value: Double): Unit = {
      if (0 <= i && i < this.data.length && 0 <= j && j < this.data(i).length)
        this.data(i)(j) = value
    }

    /**
      * 获取坐标(i,j)的值
      *
      * @param i       ith
      * @param j       jth
      * @param default 超出边界返回的值
      * @return value
      */
    def get(i: Int, j: Int, default: Double): Double = {
      if (0 <= i && i < this.data.length && 0 <= j && j < this.data(i).length)
        this.data(i)(j)
      else
        default
    }

    /**
      * 把总体运行背景输送给智能体所侦查的范围
      *
      * @param agent         目标智能体
      * @param lanesSide     左右侦查范围
      * @param patchesAhead  向前侦查的范围
      * @param patchesBehind 向后侦查的范围
      * @param position      目前所处的格子的位置
      * @return GameMap
      */
    def updateGameMap(agent: GameMap, lanesSide: Int, patchesAhead: Int, patchesBehind: Int, position: Int): GameMap = {
      for (g <- -lanesSide to lanesSide; u <- -patchesAhead until patchesBehind) {
        val j = 3 * width / 4 + u
        agent.data(g + lanesSide)(u + patchesAhead) = this.get(position + g, j, 0.0)
      }
      agent
    }

    /**
      * 把地图matrix平铺成一个vector
      *
      * @return
      */
    def flatMap(): Array[Double] = {
      val array = new Array[Double](this.data.length * this.data.head.length)
      for (i <- this.data.indices; j <- this.data(i).indices) {
        array(this.data.length * j + i) = this.data(i)(j) / 7
      }
      array
    }
  }

  class GameAgent() {
    var a_1 = 1.0
    var c_1 = 1.0
    var b_1 = 0
    var y_1: Double = 10 * math.floor(700 * math.random / 10)
    var x_1 = 0
    var f_1 = new Array[Double](60)
    this.init()

    def init(): Unit = {
      val a = (140 * math.random / 20).toInt
      val b = 1 + 0.7 * math.random
      this.x_1 = 20 * a + 4
      this.a_1 = b
      this.b_1 = a
    }

    def move(num: Boolean): Unit = {
      val b = this.y_1 - this.c_1 * this.a_1 - t_0
      if (num && 525 > y_1 && 525 <= b) {
        v_0 += 1
        w_0 += 1
        x_0 += 1
      } else if (num && 525 < this.y_1 && 525 >= b) {
        v_0 -= 1
        w_0 -= 1
        x_0 -= 1
      }
      this.y_1 = b
      val index = y_0 % this.f_1.length
      this.f_1(index) = this.c_1 * this.a_1 * 20
      val temp = 20 * this.b_1 + 4 - this.x_1
      this.x_1 = if (math.abs(temp) < 20 / 30) 20 * this.b_1 + 4 else if (0 < temp) this.x_1 + 20 / 30 else this.x_1 - 20 / 30
      if (0 > this.y_1 + 68) {
        this.y_1 = 734
        this.init()
      }
      if (700 < this.y_1 - 68) {
        this.y_1 = -34
        this.init()
      }
    }

    def i(a: Int): Unit = {
      var b = 1
      if (a == 1) b = 10
      for (i <- 0 until 15 by 10; j <- 0 until 34 by 5)
        z_0.set((this.x_1 + i) / 20, (this.y_1 + j).toInt / 10, b * this.c_1 * this.a_1.toInt)
    }

    def l(): Unit = {
      var a = 2.0
      for (b <- 1 until 5) {
        val d = z_0.get((this.x_1 + 7.5).toInt / 20, (this.y_1 - 10 * b).toInt / 10, 7)
        if (d < 7) {
          a = math.min(a, 0.5 * (b - 1))
          a = math.min(a, d / this.a_1)
        }
      }
      this.c_1 = a.toInt
    }

    def g(a: Int): Boolean = {
      val b = (this.x_1 + 7.5) / 20 + a
      val d = this.y_1 / 10
      var c = 0.5 > math.abs(this.x_1 - (20 * this.b_1 + 4))
      for (h <- (3 * -this.a_1).toInt until 4) {
        if (c) c = z_0.get(b.toInt, (d + h).toInt, 0) >= 7
      }
      if (c) this.b_1 += a
      c
    }

    def o(): Unit = {
      var a = true
      for (b <- 1 until 5) {
        val i = (this.x_1 + 7.5) / 20
        val j = (this.y_1 - 10 * b) / 10
        val default = if (a) 0 else 2
        if (a) a = z_0.get(i.toInt, j.toInt, default) >= 7
      }

      for (b <- 1 until 5) {
        val i = (this.x_1 + 7.5) / 20
        val j = (this.y_1 - 10 * b) / 10
        val default = if (a) 0 else 2
        A_0.set(i.toInt, j.toInt, default)
      }

      var index_i = (this.x_1 + 7.5) / 20 - 1
      var index_j = this.y_1 / 10
      a = 0.5 > math.abs(this.x_1 - (20 * this.b_1 + 4))
      for (b <- (3 * -this.a_1).toInt until 4) {
        if (a) a = z_0.get(index_i.toInt, (index_j + b).toInt, 0) >= 7
      }

      for (b <- (3 * -this.a_1).toInt until 4) {
        val default = if (a) 0 else 2
        A_0.set(index_i.toInt, (index_j + b).toInt, default)
      }

      index_i = (this.x_1 + 7.5) / 20 + 1
      a = 0.5 > math.abs(this.x_1 - (20 * this.b_1 + 4))
      for (b <- (3 * -this.a_1).toInt until 4) {
        if (a) a = z_0.get(index_i.toInt, (index_j + b).toInt, 0) >= 7
      }
      for (b <- (3 * -this.a_1).toInt until 4) {
        val default = if (a) 0 else 2
        A_0.set(index_i.toInt, (index_j + b).toInt, default)
      }
    }

    def j(): Unit = {
      m_0 match {
        case 1 => if (this.a_1 > 2) this.a_1 += 0.02
        case 2 => if (this.a_1 < 0) this.a_1 -= 0.02
        case 3 => if (this.g(-1)) B_0 = 0; m_0 = 0
        case 4 => if (this.g(1)) B_0 = 0; m_0 = 0
      }
    }

    def s(): Int = {
      val a = this.f_1.sum
      (a / this.f_1.length).toInt
    }
  }

}
