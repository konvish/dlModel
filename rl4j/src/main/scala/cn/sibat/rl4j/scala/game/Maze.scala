package cn.sibat.rl4j.scala.game

import java.awt.{Color, Graphics}
import javax.swing.{JFrame, WindowConstants}

class Maze extends JFrame {
  private val sx = 50 //小方格起始横坐标

  private val sy = 50 //小方格起始纵坐标

  private val maze_h = 4 //竖直方格数

  private val maze_w = 4 //水平方格数

  private val w = 40 //方格的像素大小

  private var jg: Graphics = _
  private val rect = Array(70, 70)

  /**
    * 初始化界面
    */
  def init(): Unit = {
    setTitle("Maze")
    val p = getContentPane
    setBounds(100, 100, 250, 250)
    setVisible(true)
    p.setBackground(Color.WHITE)
    setLayout(null)
    setResizable(false)
    this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    //获取专门用于在窗口界面上绘图的对象
    jg = this.getGraphics
    //绘制游戏区域
    paintComponents(jg)
  }

  /**
    * 采取动作后的环境变化
    *
    * @param action 行为
    * @return
    */
  def step(action: Int): (Array[Int], Int, Boolean) = {
    val base_action = Array(0, 0)
    action match {
      case 0 =>
        if (rect(1) > 70) base_action(1) -= w
      case 1 =>
        if (rect(1) < 190) base_action(1) += w
      case 2 =>
        if (rect(0) < 190) base_action(0) += w
      case 3 =>
        if (rect(0) > 70) base_action(0) -= w
    }

    update(base_action)

    val s_ = rect.clone()
    val origin = Array(70, 70)
    val hell1_center = Array(origin(0) + w * 2, origin(1) + w)
    val hell2_center = Array(origin(0) + w, origin(1) + w * 2)
    val oval_center = Array(origin(0) + w * 2, origin(1) + w * 2)
    var reward = 0
    var done = false
    if (s_(0) == oval_center(0) && s_(1) == oval_center(1)) {
      reward = 1
      done = true
      s_(0) = rect(0)
      s_(1) = rect(1)
    } else if ((s_(0) == hell1_center(0) && s_(1) == hell1_center(1)) || (s_(0) == hell2_center(0) && s_(1) == hell2_center(1))) {
      reward = -1
      done = true
      s_(0) = rect(0)
      s_(1) = rect(1)
    }
    (s_, reward, done)
  }

  /**
    * 绘制游戏地图
    *
    * @param g 图g
    */
  override def paintComponents(g: Graphics): Unit = {
    try { //设置线条颜色
      g.setColor(Color.BLACK)
      //绘制外层矩形框
      g.drawRect(sx, sy, maze_h * w, maze_w * w)

      for (i <- 0 until maze_h) {
        //绘制第i条竖直线
        g.drawLine(sx + (i * w), sy, sx + (i * w), sy + maze_w * w)
        //绘制第i条水平线
        g.drawLine(sx, sy + (i * w), sx + maze_h * w, sy + (i * w))
      }
      val origin = Array(70, 70)
      //hell1
      val hell1_center = Array(origin(0) + w * 2, origin(1) + w)
      g.drawRect(hell1_center(0) - 15, hell1_center(1) - 15, 30, 30)
      g.fillRect(hell1_center(0) - 15, hell1_center(1) - 15, 30, 30)
      //he112
      val hell2_center = Array(origin(0) + w, origin(1) + w * 2)
      g.drawRect(hell2_center(0) - 15, hell2_center(1) - 15, 30, 30)
      g.fillRect(hell2_center(0) - 15, hell2_center(1) - 15, 30, 30)
      g.setColor(Color.YELLOW)
      //oval
      val oval_center = Array(origin(0) + w * 2, origin(1) + w * 2)
      g.drawOval(oval_center(0) - 15, oval_center(1) - 15, 30, 30)
      g.fillOval(oval_center(0) - 15, oval_center(1) - 15, 30, 30)
      g.setColor(Color.RED)
      //agent
      g.drawRect(rect(0) - 15, rect(1) - 15, 30, 30)
      g.fillRect(rect(0) - 15, rect(1) - 15, 30, 30)
    } catch {
      case e: Exception =>
        e.printStackTrace()
    }
  }

  /**
    * 初始化状态
    */
  def reset(): Array[Int] = {
    jg.clearRect(rect(0) - 15, rect(1) - 15, 31, 31)
    rect(0) = 70
    rect(1) = 70
    paintComponents(jg)
    rect
  }

  /**
    * 更新操作后的状态
    *
    * @param move 往那个方向走
    */
  private def update(move: Array[Int]): Unit = {
    jg.clearRect(rect(0) - 15, rect(1) - 15, 31, 31)
    rect(0) += move(0)
    rect(1) += move(1)
    paintComponents(jg)
  }
}

object Maze {
  def main(args: Array[String]): Unit = {
    val maze = new Maze
    maze.init()
    maze.step(2)
    Thread.sleep(1000)
    maze.step(3)
    Thread.sleep(1000)
    maze.step(1)
  }
}
