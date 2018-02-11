package cn.sibat.test

object Test {
  def main(args: Array[String]): Unit = {
    val tempDir = System.getProperty("java.io.tmpdir")
    println(tempDir)
  }
}
