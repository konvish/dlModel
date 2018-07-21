package cn.sibat.rl4j.scala.game

import java.util.Properties

import org.json.{JSONArray, JSONObject}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object ConvnetNet {
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

  def randf(a: Double, b: Double): Double = math.random * (b - a) + a

  def randi(a: Int, b: Int): Int = math.floor(math.random * (b - a) + a).toInt

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

  class ConvLayer(opt: Properties) extends Layer {
    private var out_depth = opt.getProperty("filters").toInt
    private var sx = opt.getProperty("sx").toInt
    private var in_depth = opt.getProperty("in_depth").toInt
    private val in_sx = opt.getProperty("in_sx").toInt
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
      v.dw = zeros(v.w.length)
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

    override def backward(y: Any): Double = 0.0
  }

  class FullyConnLayer(opt: Properties) extends Layer {
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
      v.dw = zeros(v.w.length)

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

    override def backward(y: Any): Double = 0.0
  }

  class PoolLayer(opt: Properties) extends Layer {
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
      V.dw = zeros(V.w.length)
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

    override def backward(y: Any): Double = 0.0

    override def getParamsAndGrads: Array[String] = Array()
  }

  class InputLayer(opt: Properties) extends Layer {
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

    override def backward(y: Any): Double = 0.0

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

    override def getParamsAndGrads: Array[String] = Array()
  }

  class SoftmaxLayer(opt: Properties) extends Layer {
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

    override def backward(y: Any): Unit = 0.0

    override def getParamsAndGrads: Array[String] = Array()
  }

  class RegressionLayer(opt: Properties) extends Layer {
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

    override def getParamsAndGrads: Array[String] = Array()
  }

  class SVMLayer(opt: Properties) extends Layer {
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

    override def backward(y: Any): Double = 0.0

    override def getParamsAndGrads: Array[String] = Array()
  }

  class ReluLayer(opt: Properties) extends Layer {
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

    override def backward(y: Any): Double = 0.0

    override def getParamsAndGrads: Array[String] = Array()
  }

  class SigmoidLayer(opt: Properties) extends Layer {
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

    override def backward(y: Any): Double = 0.0

    override def getParamsAndGrads: Array[String] = Array()
  }

  class MaxoutLayer(opt: Properties) extends Layer {
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

    override def backward(y: Any): Double = 0.0

    override def getParamsAndGrads: Array[String] = Array()
  }

  class TanhLayer(opt: Properties) extends Layer {
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

    override def backward(y: Any): Double = 0.0

    override def getParamsAndGrads: Array[String] = Array()
  }

  class DropoutLayer(opt: Properties) extends Layer {
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
      } else {
        for (i <- 0 until N) V2.w(i) *= drop_prob
      }
      out_act = V2
      out_act
    }

    def backward(y: Int): Unit = {
      val V = in_act
      val chain_grad = out_act
      val N = V.w.length
      V.dw = new Array[Double](N)
      for (i <- 0 until N) {
        if (!dropped(i)) {
          V.dw(i) = chain_grad.dw(i)
        }
      }
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json.put("drop_prob", this.drop_prob)
      json
    }

    def fromJSON(jsonObject: JSONObject): DropoutLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this.drop_prob = jsonObject.getDouble("drop_prob")
      this
    }

    override def toString: String = toJSON.toString

    override def backward(y: Any): Double = 0.0

    override def getParamsAndGrads: Array[String] = Array()
  }

  class LocalResponseNormalizationLayer(opt: Properties) extends Layer {
    private var k = opt.getProperty("k").toDouble
    private var n = opt.getProperty("n").toInt
    private var alpha = opt.getProperty("alpha").toDouble
    private var beta = opt.getProperty("beta").toDouble
    private var out_depth = opt.getProperty("in_depth").toInt
    private var out_sx = opt.getProperty("in_sx").toInt
    private var out_sy = opt.getProperty("in_sy").toInt
    private var layer_type = "lrn"
    private var in_act: Vol = _
    private var out_act: Vol = _
    private var S_cache: Vol = _

    def forward(v: Vol, is_training: Boolean = false): Vol = {
      in_act = v
      val A = v.cloneAndZero()
      S_cache = v.cloneAndZero()
      val n2 = math.floor(n / 2).toInt
      for (x <- 0 until v.sx; y <- 0 until v.sy; i <- 0 until v.depth) {
        val ai = v.get(x, y, i)
        var den = 0.0
        for (j <- math.max(0, i - n2) to math.min(i + n2, v.depth - 1)) {
          val aa = v.get(x, y, j)
          den += aa * aa
        }
        den *= alpha / n
        den += k
        S_cache.set(x, y, i, den)
        den = math.pow(den, beta)
        A.set(x, y, i, ai / den)
      }
      out_act = A
      out_act
    }

    def backward(): Unit = {
      val V = in_act
      val N = V.w.length
      V.dw = new Array[Double](N)
      val A = out_act
      val n2 = math.floor(n / 2).toInt
      for (x <- 0 until V.sx; y <- 0 until V.sy; i <- 0 until V.depth) {
        val chain_grad = out_act.get_grad(x, y, i)
        val S = S_cache.get(x, y, i)
        val SB = math.pow(S, beta)
        val SB2 = SB * SB

        for (j <- math.max(0, i - n2) to math.min(i + n2, V.depth - 1)) {
          val aj = V.get(x, y, j)
          var g = -aj * beta * math.pow(S, beta - 1) * alpha / n * 2 * aj
          if (i == j) g += SB
          g /= SB2
          g *= chain_grad
          V.add_grad(x, y, j, g)
        }
      }
    }

    def toJSON: JSONObject = {
      val json = new JSONObject()
      json.put("out_depth", this.out_depth)
      json.put("out_sx", this.out_sx)
      json.put("out_sy", this.out_sy)
      json.put("layer_type", this.layer_type)
      json.put("k", this.k)
      json.put("n", this.n)
      json.put("alpha", this.alpha)
      json.put("beta", this.beta)
      json
    }

    def fromJSON(jsonObject: JSONObject): LocalResponseNormalizationLayer = {
      this.out_depth = jsonObject.getInt("out_depth")
      this.out_sx = jsonObject.getInt("out_sx")
      this.out_sy = jsonObject.getInt("out_sy")
      this.layer_type = jsonObject.getString("layer_type")
      this.k = jsonObject.getDouble("k")
      this.n = jsonObject.getInt("n")
      this.alpha = jsonObject.getDouble("alpha")
      this.beta = jsonObject.getDouble("beta")
      this
    }

    override def toString: String = toJSON.toString

    override def backward(y: Any): Double = 0.0

    override def getParamsAndGrads: Array[String] = Array()
  }

  abstract class Layer() {
    var out_sx: Int = _
    var out_sy: Int = _
    var out_depth: Int = _
    var layer_type: String = _
    var out_act: Vol = _

    def forward(v: Vol, is_training: Boolean): Vol

    def backward(y: Any): Double

    def getParamsAndGrads: JSONArray

    def toJSON: JSONObject

    def fromJSON(jSONObject: JSONObject)
  }

  class Net() {
    private var layers: Array[Layer] = _

    def makeLayers(defs: Array[JSONObject]): Unit = {
      require(defs.length >= 2 && defs.head.getString("layer_type") == "input")
      val new_defs = desugar(defs)
      layers = new Array[Layer](new_defs.length)
      for (i <- new_defs.indices) {
        val def_i = new_defs(i)
        if (i > 0) {
          val prev = layers(i - 1)
          def_i.put("in_sx", prev.out_sx)
          def_i.put("in_sy", prev.out_sy)
          def_i.put("in_depth", prev.out_depth)
        }
        def_i.getString("type") match {
          case "fc" => layers(i) = new FullyConnLayer(new Properties())
          case "lrn" => layers(i) = new LocalResponseNormalizationLayer(new Properties())
          case "dropout" => layers(i) = new DropoutLayer(new Properties())
          case "input" => layers(i) = new InputLayer(new Properties())
          case "softmax" => layers(i) = new SoftmaxLayer(new Properties())
          case "regression" => layers(i) = new RegressionLayer(new Properties())
          case "conv" => layers(i) = new ConvLayer(new Properties())
          case "pool" => layers(i) = new PoolLayer(new Properties())
          case "relu" => layers(i) = new ReluLayer(new Properties())
          case "sigmoid" => layers(i) = new SigmoidLayer(new Properties())
          case "tanh" => layers(i) = new TanhLayer(new Properties())
          case "maxout" => layers(i) = new MaxoutLayer(new Properties())
          case "svm" => layers(i) = new SVMLayer(new Properties())
          case _ => println("error layer type")
        }
      }
    }

    def forward(v: Vol, is_training: Boolean = false): Vol = {
      var act = layers(0).forward(v, is_training)
      for (i <- 1 until layers.length) {
        act = layers(i).forward(act, is_training)
      }
      act
    }

    def getCostLoss(v: Vol, y: Int): Double = {
      forward(v)
      val n = layers.length
      val loss = layers(n - 1).backward(y)
      loss
    }

    def backward(y: Any): Double = {
      val n = layers.length
      val loss = layers(n - 1).backward(y)
      for (i <- n - 2 to 0) {
        layers(i).backward(y)
      }
      loss
    }

    def getParamsAndGrads: JSONArray = {
      val response = new JSONArray()
      for (i <- layers.indices) {
        val layer_response = layers(i).getParamsAndGrads
        for (j <- 0 until layer_response.length()) {
          response.put(layer_response.get(j))
        }
      }
      response
    }

    def getPrediction: Int = {
      val S = layers(layers.length - 1)
      require(S.layer_type.equals("softmax"))
      val p = S.out_act.w
      p.indexOf(p.max)
    }

    def toJSON: JSONObject = {
      val jsonObject = new JSONObject()
      val jsonArray = new JSONArray()
      layers.foreach(l => jsonArray.put(l.toJSON))
      jsonObject.put("layers", jsonArray)
      jsonObject
    }

    def fromJSON(json: JSONObject): Unit = {
      val json_layers = json.getJSONArray("layers")
      layers = new Array[Layer](json_layers.length())
      for (i <- 0 until json_layers.length()) {
        val layer_json = json_layers.getJSONObject(i)
        val layer_type = layer_json.getString("layer_type")
        val layer = layer_type match {
          case "input" => new InputLayer(new Properties()).fromJSON(layer_json)
          case "relu" => new ReluLayer(new Properties()).fromJSON(layer_json)
          case "sigmoid" => new SigmoidLayer(new Properties()).fromJSON(layer_json)
          case "tanh" => new TanhLayer(new Properties()).fromJSON(layer_json)
          case "dropout" => new DropoutLayer(new Properties()).fromJSON(layer_json)
          case "conv" => new ConvLayer(new Properties()).fromJSON(layer_json)
          case "pool" => new PoolLayer(new Properties()).fromJSON(layer_json)
          case "lrn" => new LocalResponseNormalizationLayer(new Properties()).fromJSON(layer_json)
          case "softmax" => new SoftmaxLayer(new Properties()).fromJSON(layer_json)
          case "regression" => new RegressionLayer(new Properties()).fromJSON(layer_json)
          case "fc" => new FullyConnLayer(new Properties()).fromJSON(layer_json)
          case "maxout" => new MaxoutLayer(new Properties()).fromJSON(layer_json)
          case "svm" => new SVMLayer(new Properties()).fromJSON(layer_json)
        }
        layers(i) = layer
      }
    }

    private def desugar(defs: Array[JSONObject]): Array[JSONObject] = {
      val new_defs = new ArrayBuffer[JSONObject]()
      for (i <- defs.indices) {
        val def_i = defs(i)
        val layer = def_i.getString("layer_type")
        if (layer.equals("softmax") || layer.equals("svm"))
          new_defs += new JSONObject().put("type", "fc").put("num_neurons", def_i.getInt("num_classes"))
        if (layer.equals("regression"))
          new_defs += new JSONObject().put("type", "fc").put("num_neurons", def_i.getInt("num_neurons"))
        if ((layer.equals("fc") || layer.equals("conv")) && def_i.isNull("bias_pref")) {
          def_i.put("bias_pref", 0.0)
          if (!def_i.isNull("activation") && def_i.getString("activation").equals("relu"))
            def_i.put("bias_pref", 1.0)
        }
        new_defs += def_i

        if (!def_i.isNull("activation")) {
          def_i.getString("activation") match {
            case "relu" => new_defs += new JSONObject().put("type", "relu")
            case "sigmoid" => new_defs += new JSONObject().put("type", "sigmoid")
            case "tanh" => new_defs += new JSONObject().put("type", "tanh")
            case "maxout" =>
              val gs = if (def_i.isNull("group_size")) 2 else def_i.getInt("group_size")
              new_defs += new JSONObject().put("type", "maxout").put("group_size", gs)
            case _ => println("ERROR unsupported activation")
          }
        }
        if (!def_i.isNull("drop_prob") && !def_i.getString("type").equals("dropout"))
          new_defs += new JSONObject().put("type", "dropout").put("drop_prob", def_i.getDouble("drop_prob"))
      }
      new_defs.toArray
    }
  }

  class Trainer(net: Net, options: JSONObject) {
    private var learning_rate = if (options.isNull("learning_rate")) 0.01 else options.getDouble("learning_rate")
    private var l1_decay = if (options.isNull("l1_decay")) 0.0 else options.getDouble("l1_decay")
    private var l2_decay = if (options.isNull("l2_decay")) 0.0 else options.getDouble("l2_decay")
    private var batch_size = if (options.isNull("batch_size")) 1 else options.getInt("batch_size")
    private var method = if (options.isNull("method")) "sgd" else options.getString("method")

    private var momentum = if (options.isNull("momentum")) 0.9 else options.getDouble("momentum")
    private var ro = if (options.isNull("ro")) 0.95 else options.getDouble("ro")
    private var eps = if (options.isNull("eps")) 1e-6 else options.getDouble("eps")

    private var k = 0
    private var gsum: Array[Array[Double]] = _
    private var xsum: Array[Array[Double]] = _

    def train(x: Vol, y: Any): JSONObject = {
      var start = System.currentTimeMillis()
      net.forward(x, is_training = true)
      var end = System.currentTimeMillis()
      val fwd_time = end - start

      start = System.currentTimeMillis()
      val cost_loss = net.backward(y)
      var l1_decay_loss = 0.0
      var l2_decay_loss = 0.0
      end = System.currentTimeMillis()
      val bwd_time = end - start

      k += 1
      if (k % batch_size == 0) {
        val pglist = net.getParamsAndGrads
        if (gsum.length == 0 && (!method.equals("sgd") || momentum > 0.0)) {
          gsum = new Array[Array[Double]](pglist.length())
          xsum = new Array[Array[Double]](pglist.length())
          for (i <- 0 until pglist.length()) {
            val len = pglist.getJSONObject(i).getJSONArray("params").length()
            gsum(i) = new Array[Double](len)
            if (method.equals("adadelta"))
              xsum(i) = new Array[Double](len)
            else
              xsum(i) = Array[Double]()
          }
        }
        for (i <- 0 until pglist.length()) {
          val pg = pglist.getJSONObject(i)
          val p = pg.getJSONArray("params")
          val g = pg.getJSONArray("grads")

          val l2_decay_mul = if (pg.isNull("l2_decay_mul")) 1.0 else pg.getDouble("l2_decay_mul")
          val l1_decay_mul = if (pg.isNull("l1_decay_mul")) 1.0 else pg.getDouble("l1_decay_mul")
          val l2_decay_ = this.l2_decay * l2_decay_mul
          val l1_decay_ = this.l1_decay * l1_decay_mul
          val plen = p.length
          for (j <- 0 until plen) {
            l2_decay_loss += l2_decay_ * p.getDouble(j) * p.getDouble(j) / 2
            l1_decay_loss += l1_decay_ * math.abs(p.getDouble(j))
            val l1grad = l1_decay_ * (if (p.getDouble(j) > 0) 1 else -1)
            val l2grad = l2_decay_ * p.getDouble(j)

            val gij = (l2grad + l1grad + g.getDouble(j)) / batch_size
            method match {
              case "adagras" =>
                gsum(i)(j) = gsum(i)(j) + gij * gij
                val dx = -learning_rate / math.sqrt(gsum(i)(j) + eps) * gij
                p.put(j, p.getDouble(j) + dx)
              case "windowgrad" =>
                gsum(i)(j) = ro * gsum(i)(j) + (1 - ro) * gij * gij
                val dx = -learning_rate / math.sqrt(gsum(i)(j) + eps) * gij
                p.put(j, p.getDouble(j) + dx)
              case "adadelta" =>
                gsum(i)(j) = ro * gsum(i)(j) + (1 - ro) * gij * gij
                val dx = -math.sqrt((xsum(i)(j) + eps) / (gsum(i)(j) + eps)) * gij
                xsum(i)(j) = ro * xsum(i)(j) + (1 - ro) * dx * dx
                p.put(j, p.getDouble(j) + dx)
              case "nesterov" =>
                val dx = gsum(i)(j)
                gsum(i)(j) = gsum(i)(j) * momentum + learning_rate * gij
                p.put(j, p.getDouble(j) + dx)
              case _ =>
                if (momentum > 0.0) {
                  val dx = momentum * gsum(i)(j) - learning_rate * gij
                  gsum(i)(j) = dx
                  p.put(j, p.getDouble(j) + dx)
                }.asScala else
                  p.put(j, p.getDouble(j) - learning_rate * gij)
            }
            g.put(j, 0.0)
          }
        }
      }
      new JSONObject().put("fwd_time", fwd_time).put("bwd_time", bwd_time)
        .put("l2_decay_loss", l2_decay_loss).put("l1_decay_loss", l1_decay_loss)
        .put("cost_loss", cost_loss).put("softmax_loss", cost_loss)
        .put("loss", cost_loss + l1_decay_loss + l2_decay_loss)
    }
  }

  class MagicNet(data: Array[Vol], labels: Array[Double], opt: Properties) {
    private var train_ratio = opt.getProperty("train_ratio", "0.7").toDouble
    private var num_folds = opt.getProperty("num_folds", "10").toInt
    private var num_candidates = opt.getProperty("num_candidates", "50").toInt
    private var num_epochs = opt.getProperty("num_epochs", "50").toInt
    private var ensemble_size = opt.getProperty("ensemble_size", "10").toInt
    private var batch_size_min = opt.getProperty("batch_size_min", "10").toInt
    private var batch_size_max = opt.getProperty("batch_size_max", "300").toInt
    private var l2_decay_min = opt.getProperty("l2_decay_min", "-4").toInt
    private var l2_decay_max = opt.getProperty("l2_decay_max", "2").toInt
    private var learning_rate_min = opt.getProperty("learning_rate_min", "-4").toInt
    private var learning_rate_max = opt.getProperty("learning_rate_max", "0").toInt
    private var momentum_min = opt.getProperty("momentum_min", "0.9").toDouble
    private var momentum_max = opt.getProperty("momentum_max", "0.9").toDouble
    private var neurons_min = opt.getProperty("neurons_min", "5").toInt
    private var neurons_max = opt.getProperty("neurons_max", "30").toInt

    private var folds: JSONArray = _
    private var candidates: Array[JSONObject] = _
    private var evaluated_candidates: Array[JSONObject] = Array()
    private var unique_labels = arrUnique(labels)
    private var iter = 0
    private var foldix = 0
    private var finish_fold_callback: () => Unit = _
    private var finish_batch_callback: () => Unit = null

    def sampleFolds(): Unit = {
      val n = data.length
      val num_train = math.floor(train_ratio * n).toInt
      for (i <- 0 until num_folds) {
        val p = randperm(n)
        folds.put(new JSONObject().put("train_ix", p.slice(0, num_train)).put("test_ix", p.slice(num_train, n)))
      }
    }

    def sampleCandidate(): JSONObject = {
      val input_depth = data(0).w.length
      val num_classes = unique_labels.length
      val layer_defs = new ArrayBuffer[JSONObject]()
      layer_defs += new JSONObject().put("type", "input").put("out_sx", 1).put("out_sy", 1).put("out_depth", input_depth)
      val nl = weightedSample(Array(0, 1, 2, 3), Array(0.2, 0.3, 0.3, 0.2))
      for (q <- nl) {
        val ni = randi(neurons_min, neurons_max)
        val ran = new Random().nextInt(3)
        val m = Array("tanh", "maxout", "relu")
        val act = m(ran)
        if (randf(0, 1) < 0.5) {
          val dp = math.random
          layer_defs += new JSONObject().put("type", "fc").put("num_neurons", ni).put("activation", act).put("drop_prob", dp)
        } else {
          layer_defs += new JSONObject().put("type", "fc").put("num_neurons", ni).put("activation", act)
        }
      }
      layer_defs += new JSONObject().put("type", "softmax").put("num_classes", num_classes)
      val net = new Net()
      net.makeLayers(layer_defs.toArray)

      val bs = randi(batch_size_min, batch_size_max)
      val l2 = math.pow(10, randf(l2_decay_min, l2_decay_max))
      val lr = math.pow(10, randi(learning_rate_min, learning_rate_max))
      val mom = randf(momentum_min, momentum_max)
      val tp = randf(0, 1)
      val trainer_def = if (tp < 0.33) {
        new JSONObject().put("method", "adadelta").put("batch_size", bs).put("l2_decay", l2)
      } else if (tp < 0.66) {
        new JSONObject().put("method", "adagrad").put("learning_rate", lr).put("batch_size", bs).put("l2_decay", l2)
      } else {
        new JSONObject().put("method", "sgd").put("learning_rate", lr).put("momentum", mom).put("batch_size", bs).put("l2_decay", l2)
      }

      val trainer = new Trainer(net, trainer_def)

      val cand = new JSONObject()
      cand.put("acc", Array())
      cand.put("accv", 0)
      cand.put("layer_defs", layer_defs.toArray)
      cand.put("trainer_def", trainer_def)
      cand.put("net", net)
      cand.put("trainer", trainer)
      cand
    }

    def sampleCandidates(): Unit = {
      val result = new ArrayBuffer[JSONObject]()
      for (i <- 0 until num_candidates) {
        val cand = sampleCandidate()
        result += cand
      }
      candidates = result.toArray
    }

    def evalValErrors(): Array[Double] = {
      val vals = new ArrayBuffer[Double]()
      val fold = folds.getJSONObject(foldix)
      for (k <- candidates.indices) {
        val net = candidates(k).get("net").asInstanceOf[Net]
        var v = 0.0
        val test_ix = fold.getJSONArray("test_ix")
        for (q <- 0 until test_ix.length()) {
          val x = data(test_ix.get(q).asInstanceOf[Vol])
          val l = labels(test_ix.getInt(q))
          net.forward(x)
          val yhat = net.getPrediction
          val loss = if (yhat == l) 1.0 else 0.0
          v += loss
        }
        v /= test_ix.length()
        vals += v
      }
      vals.toArray
    }

    def step(): Unit = {
      iter += 1

      val fold = folds.getJSONObject(foldix)
      val train_ix = fold.getJSONArray("train_ix")
      val dataix = train_ix.getInt(randi(0, train_ix.length()))
      for (k <- candidates.indices) {
        val x = data(dataix)
        val l = labels(dataix)
        candidates(k).get("trainer").asInstanceOf[Trainer].train(x, l)
      }

      val lastiter = num_epochs * fold.getJSONArray("train_ix").length()
      if (iter >= lastiter) {
        val val_acc = evalValErrors()
        for (k <- candidates.indices) {
          val c = candidates(k)
          c.put("acc", c.getJSONArray("acc").put(val_acc(k)))
          c.put("accv", c.getDouble("accv") + val_acc(k))
        }
        iter = 0
        foldix += 1
        if (foldix >= folds.length()) {
          for (k <- candidates.indices) {
            evaluated_candidates ++= Array(candidates(k))
          }
          evaluated_candidates.sortBy(obj => obj.getDouble("accv") / obj.getJSONArray("acc").length())

          if (evaluated_candidates.length > 3 * ensemble_size) {
            evaluated_candidates = evaluated_candidates.slice(0, 3 * ensemble_size)
          }
          if (finish_batch_callback != null) {
            finish_batch_callback()
          }
          sampleCandidates()
          foldix = 0
        } else {
          for (k <- candidates.indices) {
            val c = candidates(k)
            val net = new Net()
            val layers_defs = new ArrayBuffer[JSONObject]()
            val layer = c.getJSONArray("layer_defs")
            for (i <- 0 until layer.length()) {
              layers_defs += layer.getJSONObject(i)
            }
            net.makeLayers(layers_defs.toArray)
            val trainer = new Trainer(net, c.getJSONObject("trainer_def"))
            c.put("net", net)
            c.put("trainer", trainer)
          }
        }
      }
    }

    def predict_soft(data: Vol): Vol = {
      var eval_candidates = Array[JSONObject]()
      var nv = 0
      if (evaluated_candidates.length == 0) {
        nv = candidates.length
        eval_candidates = this.candidates
      } else {
        nv = math.min(ensemble_size, evaluated_candidates.length)
        eval_candidates = evaluated_candidates
      }
      var xout: Vol = null
      var n = 0
      for (j <- 0 until nv) {
        val net = eval_candidates(j).get("net").asInstanceOf[Net]
        val x = net.forward(data)
        if (j == 0) {
          xout = x
          n = x.w.length
        } else {
          for (d <- 0 until n)
            xout.w(d) += x.w(d)
        }
      }
      for (d <- 0 until n)
        xout.w(d) /= nv
      xout
    }

    def predict(data: Vol): Int = {
      val xout = predict_soft(data)
      var predicted_label = -1
      if (xout.w.length != 0) {
        val stats = maxmin(xout.w)
        predicted_label = stats.getOrElse("maxi", 0).toInt
      }
      predicted_label
    }

    def toJSON: JSONObject = {
      val nv = math.min(ensemble_size, evaluated_candidates.length)
      val json = new JSONObject()
      val jsonNets = new JSONArray()
      for (i <- 0 until nv) {
        jsonNets.put(evaluated_candidates(i).get("net").asInstanceOf[Net].toJSON)
      }
      json.put("nets", jsonNets)
    }

    def fromJSON(json: JSONObject): Unit = {
      val nets = json.getJSONArray("nets")
      ensemble_size = nets.length()
      val result = new ArrayBuffer[JSONObject]()
      for (i <- 0 until ensemble_size) {
        val net = new Net()
        net.fromJSON(nets.getJSONObject(i))
        val dummy_candidate = new JSONObject()
        dummy_candidate.put("net", net)
        result += dummy_candidate
      }
      evaluated_candidates = result.toArray
    }

    def onFinishFold(f: () => Unit): Unit = {
      finish_fold_callback = f
    }

    def onFinishBatch(f: () => Unit): Unit = {
      finish_batch_callback = f
    }
  }

  class Vol(val sx: Int, val sy: Int, val depth: Int, val c: Double) {
    var n: Int = sx * sy * depth
    var w: Array[Double] = {
      val result = zeros(n)
      if (c.isNaN) {
        val scale = math.sqrt(1.0 / n)
        val value = randn(0.0, scale)
        result.map((d: Double) => value)
      } else {
        result.map(d => c)
      }
      result
    }
    var dw: Array[Double] = zeros(n)

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