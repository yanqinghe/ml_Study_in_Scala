import java.nio.file.{Files, Paths}

import scala.collection.JavaConverters._
import breeze.linalg.{DenseMatrix, DenseVector}
import com.sun.org.apache.xerces.internal.impl.xpath.XPath.Axis
import breeze.linalg._
import breeze.numerics._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
object KNN {
  def main(args: Array[String]): Unit = {
    val (data,labels) = loadData2Matrix("data/machinelearninginaction/Ch02/datingTestSet2.txt");
    val dataM:DenseMatrix[Double] = new DenseMatrix(data.length,data(0).length,data.transpose.flatten);
    val labelsM =DenseVector(labels);
    val (normDataM,maxVal,minVal) = autoNorm(dataM);
    val len = normDataM.rows;
    val n = (len*0.1).toInt;
    var errorCnt = 0.0;
    for(i<- 0 until(n)){
      val result = classify(normDataM(i,::),normDataM(n until len,::),labelsM(n until len),4);
      println("分类结果",result,"实际类别",labelsM(i));
      if(result!=labelsM(i)){
        errorCnt+=1;
      }
    }
    println("分类错误率",errorCnt/len)
  }

  /**
    * 分类器
    * @param input 待分类的输入向量
    * @param dataSet 数据集
    * @param lables 标签集
    * @param k k值
    */
  def classify(input:Transpose[DenseVector[Double]], dataSet:DenseMatrix[Double], lables:DenseVector[Double],k:Int): (Double) ={

    val size = dataSet.rows;
    val inputM = DenseMatrix.zeros[Double](size,input.inner.length);
    for(i <- 0 until(size))
      inputM(i,::):=input;
    val diffMat:DenseMatrix[Double] = inputM - dataSet;
    val sqDiffMat = diffMat *:* diffMat;
    val sqDis = sum(sqDiffMat(*,::));
    val distance:DenseVector[Double] = sqrt(sqDis);
    val sortedDistIndices = argsort(distance);
    val classCnt = scala.collection.mutable.HashMap.empty[Double,Int];
    for(i <- 0 until k){
      val thisLabel = lables(sortedDistIndices(i));
      classCnt.put(thisLabel,classCnt.getOrElse(thisLabel,0)+1)
    }
    classCnt.toSeq.sortWith(_._2>_._2)(0)._1;
  }

  /**
    * 读取数据文件转化为矩阵
    * @param filePath
    */
  def loadData2Matrix(filePath:String): (Array[Array[Double]], Array[Double])={
    val lines: mutable.Buffer[String] = Files.readAllLines(Paths.get(filePath)).asScala;
    val size: Int = lines.size;
    val lable = new ArrayBuffer[Double]();
    val data = lines.map(line => line.trim().split("\t").map(num => num.toDouble)).map(row => {
      lable += row(row.size - 1);
      row.take(row.size - 1);
    });
    (data.toArray, lable.toArray)
  }

  /**
    * 数据归一化
    * @param dataSet 数据集
    * @return (归一化的数据集，最大值，最小值)
    */
  def autoNorm(dataSet:DenseMatrix[Double]): (DenseMatrix[Double],Double,Double) ={
    var (maxVal,minVal) = (max(dataSet),min(dataSet));//求数据集的最大值和最小值
    var r = maxVal-minVal;
    ((dataSet:-=minVal):/=maxVal,maxVal,minVal);
  }
}
