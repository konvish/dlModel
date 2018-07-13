package cn.sibat.rl4j.scala.game

import java.util.Properties

import org.json.JSONObject

import scala.collection.mutable.ArrayBuffer

class ConvnetNet {
  private var return_v = false
  private var v_val = 0.0

  def gaussRandom(): Double = {
    if (return_v) {
      return_v = false
      v_val
    } else {
      val u = 2 * math.random - 1
      val v = 2 * math.random - 1
      val r = u * u + v * v
      if (r == 0 || r > 1)
        return gaussRandom()
      val c = math.sqrt(-2 * math.log(r) / r)
      v_val = v * c
      return_v = true
      u * c
    }
  }

  def randn(mu: Double, std: Double): Double = mu + gaussRandom() * std

  def zeros(n: Int): Array[Double] = new Array[Double](n)

  def arrContains(arr: Array[Double], elt: Double): Boolean = arr.contains(elt)

  def arrUnique(arr: Array[Double]): Array[Double] = arr.toSet.toArray

  /**
    * 返回最大值最小值以及最大最小的索引，极差
    *
    * @param w 数组
    * @return map
    */
  def maxmin(w: Array[Double]): Map[String, Double] = {
    if (w.isEmpty) return Map[String, Double]()
    val minv = w.min
    val maxv = w.max
    val mini = w.indexOf(minv)
    val maxi = w.indexOf(maxv)
    Map("maxi" -> maxi, "maxv" -> maxv, "mini" -> mini, "minv" -> minv, "dv" -> (maxv - minv))
  }

  /**
    * 随机打乱数组[0,..,n-1]
    *
    * @param n 个数
    * @return arr
    */
  def randperm(n: Int): Array[Int] = {
    val array = (0 until n).toArray
    var i = n
    while (i > -1) {
      val j = math.floor(math.random * (i + 1)).toInt
      val temp = array(i)
      array(i) = array(j)
      array(j) = temp
      i -= 1
    }
    array
  }

  /**
    * 权重抽样
    *
    * @param list  weight
    * @param probs probs.sum = 1
    * @return d
    */
  def weightedSample(list: Array[Double], probs: Array[Double]): Double = {
    val p = ConvnetNet.randf(0, 1.0)
    var cumprob = 0.0 //累积概率
    for (k <- list.indices) {
      cumprob += probs(k)
      if (p < cumprob)
        return list(k)
    }
    0.0
  }

  def forward(svol: ConvnetNet.Vol): Array[Int] = {
    Array()
  }

  def SGDTrainer(value_net: ConvnetNet, tdtrainer_options: Map[_ <: String, Double]) = {}

  def makeLayers(layer_defs: Int) = {}

}

object ConvnetNet {

  def apply: ConvnetNet = new ConvnetNet()

  def randf(a: Int, b: Double): Double = math.random * (b - a) + a

  def randi(a: Int, b: Int): Double = math.floor(math.random * (b - a) + a)

  def augment(v: Vol, crop: Int, dx: Int, dy: Int, fliplr: Boolean): Vol = {
    var W = v
    if (crop != v.sx || dx != 0 || dy != 0) {
      W = new Vol(crop, crop, v.depth, 0.0)
      for (x <- 0 until crop; y <- 0 until crop) {
        if (!(x + dx < 0 || x + dx >= v.sx || y + dy < 0 || y + dy >= v.sy))
          for (d <- 0 until v.depth)
            W.set(x, y, d, v.get(x + dx, y + dy, d))
      }
    }

    if (fliplr) {
      val W2 = W.cloneAndZero()
      for (x <- 0 until W.sx; y <- 0 until W.sy; d <- W.depth)
        W2.set(x, y, d, W.get(W.sx - x - 1, y, d))
      W = W2
    }
    W
  }

  //  def img2vol(img: BufferedImage, converGrayscale: Boolean): Unit = {
  //    val p = img.getData
  //    val W = img.getWidth
  //    val H = img.getHeight()
  //    p.getPixel()
  //  }

  class ConvLayer(opt: Properties) {
    private var out_depth = opt.getProperty("filters").toInt
    private var sx = opt.getProperty("sx").toInt
    private var in_depth = opt.getProperty("in_depth").toInt
    private var in_sx = opt.getProperty("in_sx").toInt
    private var in_sy = opt.getProperty("in_sy").toInt
    private var sy = opt.getProperty("sy", s"$sx").toInt
    private var stride = opt.getProperty("stride", "1").toInt
    private var pad = opt.getProperty("pad", "0").toInt
    private var l1_decay_mul = opt.getProperty("l1_decay_mul", "0.0").toDouble
    private var l2_decay_mul = opt.getProperty("l2_decay_mul", "1.0").toDouble
    private var out_sx = math.floor((in_sx + pad * 2 - sx) / stride + 1).toInt
    private var out_sy = math.floor((in_sy + pad * 2 - sy) / stride + 1).toInt
    private var layer_type = "conv"
    private var bias = opt.getProperty("bias_pref", "0.0").toDouble
    private var filters = (0 until out_depth).map(i => new Vol(sx, sy, in_depth, 0.0)).toArray
    private var biases = new Vol(1, 1, out_depth, bias)
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      this.in_act = v
      val A = new Vol(this.out_sx | 0, this.out_sy | 0, this.out_depth | 0, 0.0)
      val v_sx = v.sx | 0
      val v_sy = v.sy | 0
      val xy_stride = this.stride | 0
      for (d <- 0 until out_depth) {
        val f = this.filters(d)
        var x = -this.pad | 0
        var y = -this.pad | 0
        for (ay <- 0 until out_sy) {
          y += xy_stride
          x = -this.pad | 0
          for (ax <- 0 until out_sx) {
            x += xy_stride
            var a = 0.0
            for (fy <- 0 until f.sy) {
              var oy = y + fy
              for (fx <- 0 until f.sx) {
                var ox = x + fx
                if (oy >= 0 && oy < v_sy && ox >= 0 && ox < v_sx) {
                  for (fd <- 0 until f.depth)
                    a += f.w(((f.sx * fy) + fx) * f.depth + fd) * v.w(((v_sx * oy) + ox) * v.depth + fd)
                }
              }
            }
            a += this.biases.w(d)
            A.set(ax, ay, d, a)
          }
        }
      }
      this.out_act = A
      out_act
    }

    def backward(): Unit = {
      val v = this.in_act
      v.dw = ConvnetNet.apply.zeros(v.w.length)
      val v_sx = v.sx | 0
      val v_sy = v.sy | 0
      val xy_stride = this.stride | 0

      for (d <- 0 until this.out_depth) {
        val f = this.filters(d)
        var x = -pad | 0
        var y = -pad | 0
        for (ay <- 0 until out_sy) {
          y += xy_stride
          x = -pad | 0
          for (ax <- 0 until out_sx) {
            x += xy_stride
            val chain_grad = out_act.get_grad(ax, ay, d)
            for (fy <- 0 until f.sy) {
              val oy = y + fy
              for (fx <- 0 until f.sx) {
                val ox = x + fx
                if (oy >= 0 && oy < v_sy && ox >= 0 && ox < v_sx) {
                  for (fd <- 0 until f.depth) {
                    val ix1 = ((v_sx * oy) + ox) * v.depth + fd
                    val ix2 = ((f.sx * fy) + fx) * f.depth + fd
                    f.dw(ix2) += v.w(ix1) * chain_grad
                    v.dw(ix1) += f.w(ix2) * chain_grad
                  }
                }
              }
            }
            this.biases.dw(d) += chain_grad
          }
        }
      }
    }

    def getParamsAndGrads: Array[String] = {
      val response = new ArrayBuffer[String]()
      for (i <- 0 until out_depth) {
        response += s"{\"params\":${this.filters(i).w.mkString(",")},\"grads\":${this.filters(i).dw.mkString(",")},\"l2_decay_mul\":${this.l2_decay_mul},\"l1_decay_mul\":${this.l1_decay_mul}}"
      }
      response += s"{\"params\":${this.biases.w.mkString(",")},\"grads\":${this.biases.dw.mkString(",")},\"l2_decay_mul\":0.0,\"l1_decay_mul\":0.0}"
      response.toArray
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("sx", this.sx)
      json.put("sy", this.sy)
      json.put("stride", this.stride)
      json.put("in_depth", this.in_depth)
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json.put("l1_decay_mul", this.l1_decay_mul)
      json.put("l2_decay_mul", this.l2_decay_mul)
      json.put("pad", this.pad)
      json.put("filters", this.filters.map(_.toString))
      json.put("biases", this.biases.toString)
      json
    }

    def fromJSON(jsonObject: JSONObject): ConvLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this.sx = jsonObject.getInt("sx")
      this.sy = jsonObject.getInt("sy")
      this.stride = jsonObject.getInt("stride")
      this.in_depth = jsonObject.getInt("in_depth")
      this.l1_decay_mul = if (jsonObject.opt("l1_decay_mul") != null) jsonObject.getDouble("l1_decay_mul") else 1.0
      this.l2_decay_mul = if (jsonObject.opt("l2_decay_mul") != null) jsonObject.getDouble("l2_decay_mul") else 1.0
      this.pad = if (jsonObject.opt("pad") != null) jsonObject.getInt("pad") else 0
      val jsonArr = jsonObject.getJSONArray("filters")
      this.filters = new Array[Vol](jsonArr.length())
      for (i <- 0 until jsonArr.length()) {
        var v = new Vol(0, 0, 0, 0)
        v = v.fromJSON(jsonArr.getJSONObject(i))
        filters(i) = v
      }
      this.biases = new Vol(0, 0, 0, 0).fromJSON(jsonObject.getJSONObject("biases"))
      this
    }

    override def toString: String = toJSON.toString
  }

  class FullyConnLayer(opt: Properties) {
    private var out_depth = opt.getProperty("num_neurons").toInt
    private var l1_decay_mul = opt.getProperty("l1_decay_mul", "0.0").toDouble
    private var l2_decay_mul = opt.getProperty("l2_decay_mul", "1.0").toDouble
    private var num_inputs = opt.getProperty("in_sx").toInt * opt.getProperty("in_sy").toInt * opt.getProperty("in_depth").toInt
    private var out_sx = 1
    private var out_sy = 1
    private var layer_type = "fc"
    private var bias = opt.getProperty("bias_pref", "0.0").toDouble
    private var filter = (0 until out_depth).map(i => new Vol(1, 1, num_inputs, 0.0))
    private var biases = new Vol(1, 1, out_depth, bias)
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      in_act = v
      val A = new Vol(1, 1, out_depth, 0.0)
      val Vw = v.w
      for (i <- 0 until out_depth) {
        var a = 0.0
        var wi = filter(i).w
        for (j <- 0 until num_inputs)
          a += Vw(j) * wi(j)
        a += biases.w(i)
        A.w(i) = a
      }
      out_act = A
      out_act
    }

    def backward(): Unit = {
      val v = this.in_act
      v.dw = ConvnetNet.apply.zeros(v.w.length)

      for (i <- 0 until out_depth) {
        val tfi = this.filter(i)
        val chain_grad = this.out_act.dw(i)
        for (d <- 0 until num_inputs) {
          v.dw(d) += tfi.w(d) * chain_grad
          tfi.dw(d) += v.w(d) * chain_grad
        }
        this.biases.dw(i) += chain_grad
      }
    }


  }

  class Vol(val sx: Int, val sy: Int, val depth: Int, val c: Double) {
    var n: Int = sx * sy * depth
    var w: Array[Double] = {
      val result = ConvnetNet.apply.zeros(n)
      if (c.isNaN) {
        val scale = math.sqrt(1.0 / n)
        val value = ConvnetNet.apply.randn(0.0, scale)
        result.map(d => value)
      } else {
        result.map(d => c)
      }
      result
    }
    var dw: Array[Double] = ConvnetNet.apply.zeros(n)

    def get(x: Int, y: Int, d: Int): Double = {
      val ix = (sx * y + x) * depth + d
      w(ix)
    }

    def set(x: Int, y: Int, d: Int, v: Double): Unit = {
      val ix = (sx * y + x) * depth + d
      w(ix) = v
    }

    def add(x: Int, y: Int, d: Int, v: Double): Unit = {
      val ix = (sx * y + x) * depth + d
      w(ix) += v
    }

    def get_grad(x: Int, y: Int, d: Int): Double = {
      val ix = (sx * y + x) * depth + d
      dw(ix)
    }

    def set_grad(x: Int, y: Int, d: Int, v: Double): Unit = {
      val ix = (sx * y + x) * depth + d
      dw(ix) = v
    }

    def add_grad(x: Int, y: Int, d: Int, v: Double): Unit = {
      val ix = (sx * y + x) * depth + d
      dw(ix) += v
    }

    def cloneAndZero(): Vol = new Vol(sx, sy, depth, 0.0)

    override def clone: Vol = {
      val v = new Vol(sx, sy, depth, 0.0)
      v.w = w
      v
    }

    def addFrom(v: Vol): Unit = {
      w = v.w
    }

    def addFromScaled(v: Vol, a: Double): Unit = {
      w = v.w.map(_ * a)
    }

    def setConst(a: Double): Unit = {
      w = w.map(d => a)
    }

    def fromJSON(json: String): Vol = {
      val jsonObj = new JSONObject(json)
      fromJSON(jsonObj)
    }

    def fromJSON(json: JSONObject): Vol = {
      val sx = json.getInt("sx")
      val sy = json.getInt("sy")
      val depth = json.getInt("depth")
      val w = json.getJSONArray("w")
      val n = sx * sy * depth
      val v = new Vol(sx, sy, depth, 0.0)
      for (i <- 0 until n) {
        v.w(i) = w.getDouble(i)
      }
      v
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("sx", this.sx)
      json.put("sy", this.sy)
      json.put("depth", this.depth)
      json.put("w", this.w)
      json
    }

    override def toString: String = toJSON.toString
  }

}