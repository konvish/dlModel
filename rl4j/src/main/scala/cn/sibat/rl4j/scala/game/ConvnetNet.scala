package cn.sibat.rl4j.scala.game

class ConvnetNet {
  def forward(svol: ConvnetNet.Vol):Array[Int] = {Array()}

  def SGDTrainer(value_net: ConvnetNet, tdtrainer_options: Map[_ <: String, Double]) = {}

  def makeLayers(layer_defs: Int) = {}

}

object ConvnetNet{
  def randf(i: Int, d: Double) = {}

  def randi(i: Int, num_actions: Int) = {}

  case class Vol(var w: Int, i1: Int, net_inputs: Int)

}