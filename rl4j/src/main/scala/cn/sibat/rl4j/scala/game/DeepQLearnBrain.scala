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
  private val tdtrainer_options = {
    "learning_rate" -> 0.01
    , "momentum" -> 0.0
    , "batch_size" -> 64
    , "l2_decay" -> 0.01
  }

}
