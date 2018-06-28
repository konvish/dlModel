package cn.sibat.rl4j.scala.game

object RunMain {
  def main(args: Array[String]): Unit = {
    val lanesSide = 3
    val patchesAhead = 50
    val patchesBehind = 10
    val trainIterations = 10000
    val num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind)
    val num_actions = 5
    val temporal_window = 0
    val network_szie = num_inputs * temporal_window + num_actions * temporal_window + num_inputs

    val layer_defs = Map("type" -> "input", "out_sx" -> 1, "out_sy" -> 1, "out_depth" -> network_szie)

    val brain = new Deepqlearn
  }
}
