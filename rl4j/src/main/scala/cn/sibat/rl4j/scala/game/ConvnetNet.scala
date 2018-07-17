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
    private var filters = (0 until out_depth).map(i => new Vol(1, 1, num_inputs, 0.0)).toArray
    private var biases = new Vol(1, 1, out_depth, bias)
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      in_act = v
      val A = new Vol(1, 1, out_depth, 0.0)
      val Vw = v.w
      for (i <- 0 until out_depth) {
        var a = 0.0
        var wi = filters(i).w
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
        val tfi = this.filters(i)
        val chain_grad = this.out_act.dw(i)
        for (d <- 0 until num_inputs) {
          v.dw(d) += tfi.w(d) * chain_grad
          tfi.dw(d) += v.w(d) * chain_grad
        }
        this.biases.dw(i) += chain_grad
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
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json.put("num_inputs", this.num_inputs)
      json.put("l1_decay_mul", this.l1_decay_mul)
      json.put("l2_decay_mul", this.l2_decay_mul)
      json.put("filters", this.filters.map(_.toString))
      json.put("biases", this.biases.toString)
      json
    }

    def fromJSON(jsonObject: JSONObject): FullyConnLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this.num_inputs = jsonObject.getInt("num_inputs")
      this.l1_decay_mul = if (jsonObject.opt("l1_decay_mul") != null) jsonObject.getDouble("l1_decay_mul") else 1.0
      this.l2_decay_mul = if (jsonObject.opt("l2_decay_mul") != null) jsonObject.getDouble("l2_decay_mul") else 1.0
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

  class PoolLayer(opt: Properties) {
    private var sx = opt.getProperty("sx").toInt
    private var sy = opt.getProperty("sy", s"$sx").toInt
    private var in_depth = opt.getProperty("in_depth").toInt
    private var in_sx = opt.getProperty("in_sx").toInt
    private var in_sy = opt.getProperty("in_sy").toInt
    private var stride = opt.getProperty("stride", "2").toInt
    private var pad = opt.getProperty("pad", "0").toInt
    private var out_depth = in_depth
    private var out_sx = math.floor((in_sx + pad * 2 - sx) / stride + 1).toInt
    private var out_sy = math.floor((in_sy + pad * 2 - sy) / stride + 1).toInt
    private var layer_type = "pool"
    private var swithchx = new Array[Int](out_sx * out_sy * out_depth)
    private var swithchy = new Array[Int](out_sx * out_sy * out_depth)
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      in_act = v
      val A = new Vol(out_sx, out_sy, out_depth, 0.0)
      var n = 0
      for (d <- 0 until out_depth) {
        var x = -pad
        var y = -pad
        for (ax <- 0 until out_sx) {
          x += stride
          y = -pad
          for (ay <- 0 until out_sy) {
            y += stride
            var a = -99999.0
            var winx = -1
            var winy = -1
            for (fx <- 0 until sx; fy <- 0 until sy) {
              val oy = y + fy
              val ox = x + fx
              if (oy >= 0 && oy < v.sy && ox >= 0 && ox < v.sx) {
                val value = v.get(ox, oy, d)
                if (value > a) {
                  a = value
                  winx = ox
                  winy = oy
                }
              }
            }
            swithchx(n) = winx
            swithchy(n) = winy
            n += 1
            A.set(ax, ay, d, a)
          }
        }
      }
      out_act = A
      out_act
    }

    def backward(): Unit = {
      val V = in_act
      V.dw = ConvnetNet.apply.zeros(V.w.length)
      val A = out_act

      var n = 0
      for (d <- 0 until out_depth) {
        var x = -pad
        var y = -pad
        for (ax <- 0 until out_sx) {
          x += stride
          y = -pad
          for (ay <- 0 until out_sy) {
            y += stride
            val chain_grad = out_act.get_grad(ax, ay, d)
            V.add_grad(swithchx(n), swithchy(n), d, chain_grad)
            n += 1
          }
        }
      }
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json.put("sx", this.sx)
      json.put("sy", this.sy)
      json.put("stride", this.stride)
      json.put("in_depth", this.in_depth)
      json.put("pad", this.pad)
      json
    }

    def fromJSON(jsonObject: JSONObject): PoolLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this.sx = jsonObject.getInt("sx")
      this.sy = jsonObject.getInt("sy")
      this.stride = jsonObject.getInt("stride")
      this.in_depth = jsonObject.getInt("in_depth")
      this.pad = jsonObject.getInt("pad")
      this.swithchx = new Array[Int](out_sx * out_sy * out_depth)
      this.swithchy = new Array[Int](out_sx * out_sy * out_depth)
      this
    }

    override def toString: String = toJSON.toString
  }

  class InputLayer(opt: Properties) {
    private var out_depth = opt.getProperty("out_depth").toInt
    private var out_sx = opt.getProperty("sx").toInt
    private var out_sy = opt.getProperty("sy").toInt
    private var layer_type = "input"
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      in_act = v
      out_act = v
      out_act
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json
    }

    def fromJSON(jsonObject: JSONObject): InputLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this
    }

    override def toString: String = toJSON.toString
  }

  class SoftmaxLayer(opt: Properties) {
    private var num_inputs = opt.getProperty("in_sx").toInt * opt.getProperty("in_sy").toInt * opt.getProperty("in_depth").toInt
    private var out_depth = num_inputs
    private var out_sx = 1
    private var out_sy = 1
    private var layer_type = "softmax"
    private val es = new Array[Double](out_depth)
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      in_act = v
      val A = new Vol(1, 1, out_depth, 0.0)
      val as = v.w
      val amax = v.w.max
      var esum = 0.0
      for (i <- 0 until out_depth) {
        val e = math.exp(as(i) - amax)
        esum += e
        es(i) = e
      }

      for (i <- 0 until out_depth) {
        es(i) /= esum
        A.w(i) = es(i)
      }

      out_act = A
      out_act
    }

    def backward(y: Int): Double = {
      val x = in_act
      x.dw = new Array[Double](x.w.length)
      for (i <- 0 until out_depth) {
        val indicator = if (i == y) 1.0 else 0.0
        val mul = -(indicator - this.es(i))
        x.dw(i) = mul
      }
      -math.log(es(y))
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json.put("num_inputs", this.num_inputs)
      json
    }

    def fromJSON(jsonObject: JSONObject): SoftmaxLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this.num_inputs = jsonObject.getInt("num_inputs")
      this
    }

    override def toString: String = toJSON.toString
  }

  class RegressionLayer(opt: Properties) {
    private var num_inputs = opt.getProperty("in_sx").toInt * opt.getProperty("in_sy").toInt * opt.getProperty("in_depth").toInt
    private var out_depth = num_inputs
    private var out_sx = 1
    private var out_sy = 1
    private var layer_type = "regression"
    private val es = new Array[Double](out_depth)
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      in_act = v
      out_act = v
      out_act
    }

    def backward(y: Any): Double = {
      val x = in_act
      x.dw = new Array[Double](x.w.length)
      var loss = 0.0
      y match {
        case y_arr: Array[Double] =>
          for (i <- 0 until out_depth) {
            val dy = x.w(i) - y_arr(i)
            x.dw(i) = dy
            loss += 0.5 * dy * dy
          }
        case y_d: Double =>
          val dy = x.w(0) - y_d
          x.dw(0) = dy
          loss += 0.5 * dy * dy
        case _ =>
      }
      loss
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json.put("num_inputs", this.num_inputs)
      json
    }

    def fromJSON(jsonObject: JSONObject): RegressionLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this.num_inputs = jsonObject.getInt("num_inputs")
      this
    }

    override def toString: String = toJSON.toString
  }

  class SVMLayer(opt: Properties) {
    private var num_inputs = opt.getProperty("in_sx").toInt * opt.getProperty("in_sy").toInt * opt.getProperty("in_depth").toInt
    private var out_depth = num_inputs
    private var out_sx = 1
    private var out_sy = 1
    private var layer_type = "svm"
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      in_act = v
      out_act = v
      out_act
    }

    def backward(y: Int): Double = {
      val x = in_act
      x.dw = new Array[Double](x.w.length)
      val yscore = x.w(y)
      val margin = 1.0
      var loss = 0.0
      for (i <- 0 until out_depth) {
        if (y != i) {
          val ydiff = -yscore + x.w(i) + margin
          if (ydiff > 0) {
            x.dw(i) += 1
            x.dw(i) -= 1
            loss += ydiff
          }
        }
      }
      loss
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json.put("num_inputs", this.num_inputs)
      json
    }

    def fromJSON(jsonObject: JSONObject): SVMLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this.num_inputs = jsonObject.getInt("num_inputs")
      this
    }

    override def toString: String = toJSON.toString
  }

  class ReluLayer(opt: Properties) {
    private var out_depth = opt.getProperty("in_depth").toInt
    private var out_sx = opt.getProperty("in_sx").toInt
    private var out_sy = opt.getProperty("in_sy").toInt
    private var layer_type = "relu"
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      in_act = v
      val V2 = v.clone
      val v2w = V2.w.map(d => if (d < 0) 0 else d)
      out_act = V2
      out_act
    }

    def backward(y: Int): Unit = {
      val V = in_act
      val V2 = out_act
      val N = V.w.length
      V.w = new Array[Double](N)
      for (i <- 0 until N) {
        if (V2.w(i) <= 0)
          V.dw(i) = 0
        else
          V.dw(i) = V2.dw(i)
      }
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json
    }

    def fromJSON(jsonObject: JSONObject): ReluLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this
    }

    override def toString: String = toJSON.toString
  }

  class SigmoidLayer(opt: Properties) {
    private var out_depth = opt.getProperty("in_depth").toInt
    private var out_sx = opt.getProperty("in_sx").toInt
    private var out_sy = opt.getProperty("in_sy").toInt
    private var layer_type = "sigmoid"
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      in_act = v
      val V2 = v.cloneAndZero()
      val N = v.w.length
      val vw = v.w
      val v2w = V2.w
      for (i <- 0 until N) {
        v2w(i) = 1.0 / (1.0 + math.exp(-vw(i)))
      }
      out_act = V2
      out_act
    }

    def backward(y: Int): Unit = {
      val V = in_act
      val V2 = out_act
      val N = V.w.length
      V.w = new Array[Double](N)
      for (i <- 0 until N) {
        if (V2.w(i) <= 0)
          V.dw(i) = 0
        else
          V.dw(i) = V2.dw(i)
      }
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json
    }

    def fromJSON(jsonObject: JSONObject): SigmoidLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this
    }

    override def toString: String = toJSON.toString
  }

  class MaxoutLayer(opt: Properties) {
    private var group_size = opt.getProperty("group_size", "2").toInt
    private var out_depth = opt.getProperty("in_depth").toInt / group_size
    private var out_sx = opt.getProperty("in_sx").toInt
    private var out_sy = opt.getProperty("in_sy").toInt
    private var layer_type = "maxout"
    private var switches = new Array[Int](out_sx * out_sy * out_depth)
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      in_act = v
      val N = out_depth
      val V2 = new Vol(out_sx, out_sy, out_depth, 0.0)
      if (out_sx == 1 && out_sy == 1) {
        for (i <- 0 until N) {
          val ix = i * group_size
          var a = v.w(ix)
          var ai = 0
          for (j <- 1 until group_size) {
            val a2 = v.w(ix + j)
            if (a2 > a) {
              a = a2
              ai = j
            }
          }
          V2.w(i) = a
          switches(i) = ix + ai
        }
      } else {
        var n = 0
        for (x <- 0 until v.sx; y <- 0 until v.sy; i <- 0 until N) {
          val ix = i * group_size
          var a = v.get(x, y, ix)
          var ai = 0
          for (j <- 1 until group_size) {
            val a2 = v.get(x, y, ix + j)
            if (a2 > a) {
              a = a2
              ai = j
            }
          }
          V2.set(x, y, i, a)
          switches(n) = ix + ai
          n += 1
        }
      }
      out_act = V2
      out_act
    }

    def backward(y: Int): Unit = {
      val V = in_act
      val V2 = out_act
      val N = out_depth
      V.dw = new Array[Double](V.w.length)
      if (out_sx == 1 && out_sy == 1) {
        for (i <- 0 until N) {
          val chain_grad = V2.dw(i)
          V.dw(switches(i)) = chain_grad
        }
      } else {
        var n = 0
        for (x <- 0 until V2.sx; y <- 0 until V2.sy; i <- 0 until N) {
          val chain_grad = V2.get_grad(x, y, i)
          V.set_grad(x, y, switches(n), chain_grad)
          n += 1
        }
      }
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json.put("group_size", this.group_size)
      json
    }

    def fromJSON(jsonObject: JSONObject): MaxoutLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this.group_size = jsonObject.getInt("group_size")
      this.switches = new Array[Int](group_size)
      this
    }

    override def toString: String = toJSON.toString
  }

  class TanhLayer(opt: Properties) {
    private var out_depth = opt.getProperty("in_depth").toInt
    private var out_sx = opt.getProperty("in_sx").toInt
    private var out_sy = opt.getProperty("in_sy").toInt
    private var layer_type = "tanh"
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol = {
      in_act = v
      val V2 = v.cloneAndZero()
      val N = v.w.length
      for (i <- 0 until N) {
        V2.w(i) = math.tanh(v.w(i))
      }
      out_act = V2
      out_act
    }

    def backward(y: Int): Unit = {
      val V = in_act
      val V2 = out_act
      val N = V.w.length
      V.dw = new Array[Double](N)
      for (i <- 0 until N) {
        val v2wi = V2.w(i)
        V.dw(i) = (1.0 - v2wi * v2wi) * V2.dw(i)
      }
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json
    }

    def fromJSON(jsonObject: JSONObject): TanhLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this
    }

    override def toString: String = toJSON.toString
  }

  class DropoutLayer(opt: Properties) {
    private var out_depth = opt.getProperty("in_depth").toInt
    private var out_sx = opt.getProperty("in_sx").toInt
    private var out_sy = opt.getProperty("in_sy").toInt
    private var layer_type = "dropout"
    private var drop_prob = opt.getProperty("drop_prob", "0.5").toDouble
    private var dropped = new Array[Boolean](out_sx * out_sy * out_depth)
    private var in_act: Vol = _
    private var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean = false): Vol = {
      in_act = v
      val V2 = v.clone()
      val N = v.w.length
      if (is_training) {
        for (i <- 0 until N) {
          if (math.random < drop_prob) {
            V2.w(i) = 0
            dropped(i) = true
          } else {
            dropped(i) = false
          }
        }
      }
      out_act = V2
      out_act
    }

    def backward(y: Int): Unit = {
      val V = in_act
      val V2 = out_act
      val N = V.w.length
      V.dw = new Array[Double](N)
      for (i <- 0 until N) {
        val v2wi = V2.w(i)
        V.dw(i) = (1.0 - v2wi * v2wi) * V2.dw(i)
      }
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json
    }

    def fromJSON(jsonObject: JSONObject): DropoutLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this
    }

    override def toString: String = toJSON.toString
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