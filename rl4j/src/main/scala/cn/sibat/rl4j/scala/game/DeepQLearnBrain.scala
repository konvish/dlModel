package cn.sibat.rl4j.scala.game

import java.util.Properties

class DeepQLearnBrain(num_states: Int,
                      num_actions: Int,
                      opt: Properties
                     ) {
  private val temporal_window = opt.getProperty("temporal_window", "1").toInt
  private val experience_size = opt.getProperty("experience_size", "30000").toInt
  private val gamma = opt.getProperty("gamma", "0.8").toDouble
  private val learning_steps_total = opt.getProperty("learning_steps_total", "100000").toInt
  private val learning_steps_burnin = opt.getProperty("learning_steps_burnin", "3000").toInt
  private val epsilon_min = opt.getProperty("epsilon_min", "0.05").toDouble
  private val epsilon_test_time = opt.getProperty("epsilon_test_time", "0.01").toDouble
  private val random_action_distribution = opt.getProperty("random_action_distribution", "").split(",").map(_.toDouble)
  private val net_inputs = num_states * this.temporal_window + num_actions * this.temporal_window + num_states
  private val window_size = math.max(temporal_window, 2)
  private val state_window = new Array[Int](window_size)
  private val action_window = new Array[Int](window_size)
  private val reward_dindow = new Array[Int](window_size)
  private val net_window = new Array[Int](window_size)
  private val layer_defs = if (opt.containsKey("layer_defs")) {
    1
  } else {
    0
  }

  private val value_net = new ConvnetNet()
  value_net.makeLayers(layer_defs)
  private val tdtrainer_options = if (opt.containsKey("tdtrainer_options")) {
    opt.getProperty("tdtrainer_options").toMap
  } else Map(
    "learning_rate" -> 0.01
    , "momentum" -> 0.0
    , "batch_size" -> 64
    , "l2_decay" -> 0.01
  )

  val tdtrainer = new ConvnetNet().SGDTrainer(value_net, tdtrainer_options)
  val experience = Array()
  val age = 0
  var forward_passes = 0
  var epsilon = 1.0
  val latest_reward = 0
  var last_input_array = Array[Double]()
  val average_reward_window = (1000, 10)
  val average_loss_window = (1000, 10)
  val learning = true

  def random_action(): Unit = {
    if (random_action_distribution.length == 0) {
      ConvnetNet.randi(0, num_actions)
    } else {
      val p = ConvnetNet.randf(0, 1.0)
      var cumprob = 0.0
      for (k <- 0 until num_actions) {
        cumprob += random_action_distribution(k)
        if (p < cumprob)
          k
      }
    }
  }

  def policy(s: Array[Double]): Map[String, Int] = {
    val svol = new ConvnetNet.Vol(1, 1, net_inputs)
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
    Map("action" -> maxk, "value" -> maxval)
  }

  def getNetInput(xt: Array[Double]): Array[Double] = {
    var w = Array[Double]()
    w = w ++ xt
    val n = window_size
    for (k <- 0 until temporal_window) {
      w = w ++ Array(state_window(n - 1 - k))
      val actionlofk = new Array[Double](num_actions)
      actionlofk(action_window(n-1-k)) = 1.0 * num_states
      w = w ++ actionlofk
    }
    w
  }

  def forward(input_array:Array[Double]): Unit ={
    forward_passes += 1
    last_input_array = input_array
    var action = _
    if (forward_passes > temporal_window){
      val net_input = getNetInput(input_array)
      if (learning){
        epsilon = math.min(1.0,math.max(epsilon_min,1.0-(age - learning_steps_burnin)/(learning_steps_total - learning_steps_burnin)))
      }else{
        epsilon = epsilon_test_time
      }
      val rf = ConvnetNet.randf(0,1)
      if (rf < epsilon){
        action = random_action()
      }else{
        val maxact = policy(net_input)
        action = maxact.get("action")
      }
    }else{

    }
  }
}
