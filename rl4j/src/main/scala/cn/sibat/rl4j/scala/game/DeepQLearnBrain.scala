package cn.sibat.rl4j.scala.game

import org.json.JSONObject

import scala.collection.mutable.ArrayBuffer

/**
  * 一个agent在state0执行了action0
  * 环境评估反馈reward0，更新环境为新的转态state1
  */
class Experience() {
  var state0: Array[Double] = _
  var action0: Int = _
  var reward0: Double = _
  var state1: Array[Double] = _

  def init(state_0: Array[Double], action0: Int, reward0: Double, state1: Array[Double]): Unit = {
    this.state0 = state_0
    this.action0 = action0
    this.reward0 = reward0
    this.state1 = state1
  }
}

/**
  * 接收新的input和reward，让outputs 最大化reward
  *
  * @param num_states  states
  * @param num_actions action
  * @param opt         配置
  */
class DeepQLearnBrain(num_states: Int,
                      num_actions: Int,
                      opt: JSONObject
                     ) {
  //记录的偏移窗口
  private val temporal_window = if (opt.isNull("temporal_window")) 1 else opt.getInt("temporal_window")
  private val experience_size = if (opt.isNull("experience_size")) 30000 else opt.getInt("experience_size")
  private val start_learn_threshold = if (opt.isNull("start_learn_threshold")) math.floor(math.min(this.experience_size * 0.1, 1000)).toInt else opt.getInt("experience_size")
  private val gamma = if (opt.isNull("gamma")) 0.8 else opt.getDouble("gamma")
  private val learning_steps_total = if (opt.isNull("learning_steps_total")) 100000 else opt.getInt("learning_steps_total")
  private val learning_steps_burnin = if (opt.isNull("learning_steps_burnin")) 3000 else opt.getInt("learning_steps_burnin")
  private val epsilon_min = if (opt.isNull("epsilon_min")) 0.05 else opt.getDouble("epsilon_min")
  private val epsilon_test_time = if (opt.isNull("epsilon_test_time")) 0.01 else opt.getDouble("epsilon_test_time")
  private val random_action_distribution = if (opt.isNull("random_action_distribution")) Array() else opt.getString("random_action_distribution").split(",").map(_.toDouble)
  private val net_inputs = num_states * this.temporal_window + num_actions * this.temporal_window + num_states
  private val window_size = math.max(temporal_window, 2)
  private var state_window = new Array[Int](window_size)
  private var action_window = new Array[Int](window_size)
  private var reward_dindow = new Array[Int](window_size)
  private var net_window = new Array[Array[Double]](window_size)
  private val layer_defs = if (opt.isNull("layers_defs")) {
    val result = new ArrayBuffer[JSONObject]()
    result += new JSONObject().put("type", "input").put("out_sx", 1).put("out_sy", 1).put("out_depth", net_inputs)
    if (!opt.isNull("hidden_layer_sizes")) {
      val h1 = opt.getJSONArray("hidden_layer_sizes")
      for (k <- 0 until h1.length()) {
        result += new JSONObject().put("type", "fc").put("num_neurons", h1.getInt(k)).put("activation", "relu")
      }
    }
    result += new JSONObject().put("type", "regression").put("num_neurons", num_actions)
    result.toArray
  } else {
    val result = new ArrayBuffer[JSONObject]()
    val layers = opt.getJSONArray("layer_defs")
    for (k <- 0 until layers.length()) {
      result += layers.getJSONObject(k)
    }
    result.toArray
  }

  private val value_net = new ConvnetNet.Net()
  value_net.makeLayers(layer_defs)
  private val tdtrainer_options = if (opt.isNull("tdtrainer_options"))
    new JSONObject().put("learning_rate", 0.01).put("momentum", 0.0).put("batch_size", 64).put("l2_decay", 0.01)
  else opt.getJSONObject("tdtrainer_options")

  val tdtrainer = new ConvnetNet.Trainer(value_net, tdtrainer_options)
  val experience = new ArrayBuffer[Experience]()
  var age = 0
  var forward_passes = 0
  var epsilon = 1.0
  var latest_reward = 0
  var last_input_array: Array[Double] = Array[Double]()
  val average_reward_window = new CNNUtil.Window(1000, 10)
  val average_loss_window = new CNNUtil.Window(1000, 10)
  var learning = true

  def random_action(): Int = {
    var result = 0
    if (random_action_distribution.length == 0) {
      result = ConvnetNet.randi(0, num_actions)
    } else {
      val p = ConvnetNet.randf(0, 1.0)
      var cumprob = 0.0
      for (k <- 0 until num_actions) {
        cumprob += random_action_distribution(k)
        if (p < cumprob)
          result = k
      }
    }
    result
  }

  def policy(s: Array[Double]): JSONObject = {
    val svol = new ConvnetNet.Vol(1, 1, net_inputs, 0.0)
    svol.w = s
    val action_values = value_net.forward(svol)
    var maxk = 0
    var maxval = action_values.w(0)
    for (k <- 1 until num_actions) {
      if (action_values.w(k) > maxval) {
        maxk = k
        maxval = action_values.w(k)
      }
    }
    new JSONObject().put("action", maxk).put("value", maxval)
  }

  def getNetInput(xt: Array[Double]): Array[Double] = {
    var w = Array[Double]()
    w = w ++ xt
    val n = window_size
    for (k <- 0 until temporal_window) {
      w = w ++ Array(state_window(n - 1 - k))
      val action1ofk = new Array[Double](num_actions)
      action1ofk(action_window(n - 1 - k)) = 1.0 * num_states
      w = w ++ action1ofk
    }
    w
  }

  def forward(input_array: Array[Double]): Int = {
    forward_passes += 1
    last_input_array = input_array
    var action = 0
    var net_input = Array[Double]()
    if (forward_passes > temporal_window) {
      net_input = getNetInput(input_array)
      if (learning) {
        epsilon = math.min(1.0, math.max(epsilon_min, 1.0 - (age - learning_steps_burnin) / (learning_steps_total - learning_steps_burnin)))
      } else {
        epsilon = epsilon_test_time
      }
      val rf = ConvnetNet.randf(0, 1)
      if (rf < epsilon) {
        action = random_action()
      } else {
        val maxact = policy(net_input)
        action = maxact.getInt("action")
      }
    } else {
      net_input = Array[Double]()
      action = random_action()
    }
    net_window = net_window.tail
    net_window ++= Array(net_input)
    state_window = state_window.tail
    state_window ++= input_array
    action_window = action_window.tail
    action_window = action_window ++ Array(action)
    action
  }

  def backward(reward: Int): Unit = {
    latest_reward = reward
    average_reward_window.add(reward)
    reward_dindow = reward_dindow.tail
    reward_dindow ++= Array(reward)

    if (learning) {
      age += 1
      if (forward_passes > temporal_window + 1) {
        val e = new Experience()
        val n = window_size
        e.init(net_window(n - 2), action_window(n - 2), reward_dindow(n - 2), net_window(n - 1))
        if (experience.length < experience_size) {
          experience += e
        } else {
          val ri = ConvnetNet.randi(0, this.experience_size)
          experience(ri) = e
        }
      }

      if (experience.length > start_learn_threshold) {
        var avcost = 0.0
        for (k <- 0 until tdtrainer.batch_size) {
          val re = ConvnetNet.randi(0, this.experience_size)
          val e = experience(re)
          val x = new ConvnetNet.Vol(1, 1, net_inputs, 0.0)
          x.w = e.state0
          val maxact = policy(e.state1)
          val r = e.reward0 + gamma * maxact.getDouble("value")
          val ystruct = new JSONObject().put("dim", e.action0).put("val", r)
          val loss = tdtrainer.train(x, ystruct)
          avcost += loss.getDouble("loss")
        }
        avcost = avcost / tdtrainer.batch_size
        average_loss_window.add(avcost)
      }
    }
  }
}
