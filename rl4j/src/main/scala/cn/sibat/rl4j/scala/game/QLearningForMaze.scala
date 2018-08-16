package cn.sibat.rl4j.scala.game

import scala.collection.mutable
import scala.util.Random

class QLearningForMaze(actions: Array[Int]) {
  val learning_rate = 0.01
  val gama = 0.9
  val epsilon = 0.9
  val q_table = new mutable.HashMap[String, Array[Double]]()

  /**
    * 选择q值最大的行为
    *
    * @param observation state
    * @return action
    */
  def choose_action(observation: String): Int = {
    var action = 0
    check_state_exist(observation)
    val ran = new Random()
    if (math.random < epsilon) {
      val state_action = q_table.getOrElse(observation, new Array[Double](actions.length))
      if (state_action.max == 0.0) {
        action = actions(ran.nextInt(actions.length))
      } else
        action = state_action.indexOf(state_action.max)
    } else {
      action = actions(ran.nextInt(actions.length))
    }
    action
  }

  /**
    * q-learning
    *
    * @param s   当前状态
    * @param a   行为
    * @param r   反馈
    * @param s_1 下一行为
    */
  def learn(s: String, a: Int, r: Int, s_1: String): Unit = {
    check_state_exist(s_1)
    val array = q_table.getOrElse(s, Array())
    val q_predict = array(a)
    var q_target = 0.0
    if (!s_1.equals("-1,-1")) {
      q_target = r + q_table.getOrElse(s_1, Array()).max
    } else
      q_target = r
    array(a) += learning_rate * (q_target - q_predict)
    q_table.update(s, array)
  }

  /**
    * 检查当前的状态是否存在于q_table中
    *
    * @param state state
    */
  def check_state_exist(state: String): Unit = {
    if (!q_table.contains(state))
      q_table += ((state, new Array[Double](actions.length)))
  }
}

object QLearningForMaze {
  def main(args: Array[String]): Unit = {
    val n_actions = Array("u", "d", "l", "r")
    val maze = new Maze()
    maze.init()
    val rl = new QLearningForMaze(n_actions.indices.toArray)
    for (i <- 0 to 100) {
      var observation = maze.reset()
      var temp = true
      while (temp) {
        val action = rl.choose_action(observation.mkString(","))
        val next = maze.step(action)
        Thread.sleep(50)
        rl.learn(observation.mkString(","), action, next._2, next._1.mkString(","))
        observation = next._1
        if (next._3)
          temp = false
      }
      Thread.sleep(100)
      temp = true
    }
    rl.q_table.foreach(t => {
      println(t._1 + ":" + t._2.mkString(","))
    })
    System.exit(0)
  }
}