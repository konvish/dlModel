package scala

import java.io.{File, IOException}
import java.nio.charset.Charset
import java.nio.file.Files
import java.util.{Collections, Random, Map, LinkedList, HashMap, List}

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ArrayBuffer

class CharacterIterator extends DataSetIterator {
  private var validCharacters: Array[Char] = _
  private var charToIdxMap: Map[Character, Int] = _
  private var fileCharacters: Array[Char] = _
  private var exampleLength = 0
  private var miniBatchSize = 0
  private var rng: Random = _
  private var exampleStartOffsets: LinkedList[Int] = new LinkedList[Int]()

  @throws[IOException]
  def this(textFilePath: String, textFileEncoding: Charset, miniBatchSize: Int, exampleLength: Int
           , validCharacters: Array[Char], rng: Random) {
    this()
    if (!new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): ")
    if (miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)")
    this.validCharacters = validCharacters
    this.exampleLength = exampleLength
    this.miniBatchSize = miniBatchSize
    this.rng = rng

    charToIdxMap = new HashMap[Character, Int]()

    for (i <- validCharacters.indices) charToIdxMap.put(validCharacters(i), i)

    val newLineValid = charToIdxMap.containsKey('\n')
    val lines = Files.readAllLines(new File(textFilePath).toPath, textFileEncoding)
    var maxSize = lines.size()
    for (i <- 0 until lines.size()) {
      maxSize += lines.get(i).length
    }

    val characters = new Array[Char](maxSize)
    var currIdx = 0
    for (i <- 0 until lines.size()) {
      val thisLine = lines.get(i).toCharArray
      thisLine.foreach(c => {
        if (charToIdxMap.containsKey(c)) {
          characters(currIdx) = c
          currIdx += 1
        }
      })
      if (newLineValid) {
        characters(currIdx) = '\n'
        currIdx += 1
      }
    }

    if (currIdx == characters.length) {
      fileCharacters = characters
    } else {
      fileCharacters = characters.slice(0, currIdx)
    }

    if (exampleLength >= fileCharacters.length) throw new IllegalArgumentException("exampleLength=" + exampleLength + " cannot exceed number of valid characters in file (" + fileCharacters.length + ")")
    val nRemoved = maxSize - fileCharacters.length
    println("Loaded and converted file: " + fileCharacters.length + " valid characters of " + maxSize + " total characters (" + nRemoved + " removed)")
    initializeOffsets()
  }

  private def initializeOffsets(): Unit = {
    val nMinibatchesPerEpoch = (fileCharacters.length - 1) / exampleLength - 2
    for (i <- 0 until nMinibatchesPerEpoch) {
      exampleStartOffsets.add(i * exampleLength)
    }
    Collections.shuffle(exampleStartOffsets, rng)
  }

  def convertIndexToCharacter(idx: Int): Char = validCharacters(idx)

  def convertCharToIndex(char: Char): Int = charToIdxMap.get(char)

  def getRandomCharacter: Char = validCharacters((rng.nextDouble() * validCharacters.length).toInt)

  override def cursor(): Int = totalExamples() - exampleStartOffsets.size()

  override def next(num: Int): DataSet = {
    if (exampleStartOffsets.size() == 0) throw new NoSuchElementException
    val currMiniBatchSize = math.min(num, exampleStartOffsets.size())

    val input = Nd4j.create(Array(currMiniBatchSize, validCharacters.length, exampleLength), 'f')
    val labels = Nd4j.create(Array(currMiniBatchSize, validCharacters.length, exampleLength), 'f')

    for (i <- 0 until currMiniBatchSize) {
      val startIdx = exampleStartOffsets.removeFirst()
      val endIdx = startIdx + exampleLength
      var currCharIdx = charToIdxMap.get(fileCharacters(startIdx))
      var c = 0
      for (j <- startIdx + 1 until endIdx) {
        c += 1
        val nextCharIdx = charToIdxMap.get(fileCharacters(j))
        input.putScalar(Array(i, currCharIdx, c), 1.0)
        labels.putScalar(Array(i, nextCharIdx, c), 1.0)
        currCharIdx = nextCharIdx
      }
    }
    new DataSet(input, labels)
  }

  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = throw new UnsupportedOperationException("Not implemented")

  override def getPreProcessor: DataSetPreProcessor = throw new UnsupportedOperationException("Not implemented")

  override def totalOutcomes(): Int = validCharacters.length

  override def getLabels: java.util.List[String] = {
    throw new UnsupportedOperationException("Not implemented")
  }

  override def inputColumns(): Int = validCharacters.length

  override def resetSupported(): Boolean = true

  override def asyncSupported(): Boolean = true

  override def batch(): Int = miniBatchSize

  override def reset(): Unit = {
    exampleStartOffsets.clear()
    initializeOffsets()
  }

  override def totalExamples(): Int = (fileCharacters.length - 1) / miniBatchSize - 2

  override def numExamples(): Int = totalExamples()

  override def next(): DataSet = next(miniBatchSize)

  override def hasNext: Boolean = exampleStartOffsets.size() > 0

  override def remove(): Unit = throw new UnsupportedOperationException
}

object CharacterIterator {

  def getMinimalCharacterSet: Array[Char] = {
    val validChars = new ArrayBuffer[Char]()
    validChars ++= ('a' to 'z')
    validChars ++= ('A' to 'Z')
    validChars ++= ('0' to '9')
    val temp = Array('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')
    validChars ++= temp
    validChars.toArray
  }

  def getDefaultCharacterSet: Array[Char] = {
    val validChars = new ArrayBuffer[Char]()
    validChars ++= getMinimalCharacterSet
    val additionalChars = Array('@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_', '\\', '|', '<', '>')
    validChars ++= additionalChars
    validChars.toArray
  }
}
