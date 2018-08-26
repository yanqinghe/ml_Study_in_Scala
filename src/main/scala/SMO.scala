import java.nio.file.{Files, Path, Paths}

import breeze.linalg.{DenseMatrix, DenseVector, Transpose}
import breeze.linalg._
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import util.control.Breaks._
import breeze.numerics.exp;
object SMO {
  def main(args: Array[String]): Unit = {
    val (data, lable) = loadDataSet("data/machinelearninginaction/Ch06/testSet.txt");
//    val (b1, alphas1) = somSimple(data, lable, 0.6, 0.001, 40);
//    val (b, alphas) = smo(data, lable, 0.6, 0.001, 40);
//    val d = DenseMatrix.zeros[Double](2,3);
//    val v = DenseVector.zeros[Double](3);
//    println(d(1,::)+v.t);
//    val v2 =   DenseVector[Double](1.1,1.1,1.1);
//    println(v:=v2);
//    val w = calcWs(alphas, data, lable);
//    calcWs(alphas1, data, lable);
    testRBF()
  }

  def loadDataSet(filePath: String): (Array[Array[Double]], Array[Double]) = {
    val lines: mutable.Buffer[String] = Files.readAllLines(Paths.get(filePath)).asScala;
    val size: Int = lines.size;
    //    val data=new Array [Double] (size);
    val lable = new ArrayBuffer[Double]();
    val data = lines.map(line => line.trim().split("\t").map(num => num.toDouble)).map(row => {
      lable += row(row.size - 1);
      row.take(row.size - 1);
    });
    (data.toArray, lable.toArray)
  }

  def selectJrand(i: Int, m: Int): Int = {
    var j = i;
    while (j == i) {
      j = (new util.Random).nextInt(m)
    }
    j;
  }

  def clipAlpha(aj: Double, H: Double, L: Double): Double = {
    var t = aj;
    if (aj > H) t = H;
    else if (aj < L) t = L;
    t;
  }

  def somSimple(dataMatIn: Array[Array[Double]], classLabels: Array[Double], C: Double, toler: Double, maxIter: Int): (Double, DenseVector[Double]) = {
    //生成数据矩阵和标签列向量
    var dataFlatten = dataMatIn.transpose.flatten;
    val dataMatrix = new DenseMatrix(dataMatIn.length, dataMatIn(0).length, dataFlatten);
    val labelMatrix = DenseVector(classLabels);
    //初始化变量b
    var b: Double = 0.0;
    //获取数据矩阵的大小
    val (m, n) = (dataMatrix.rows, dataMatrix.cols);
    //初始化a列向量
    val alphas = DenseVector.zeros[Double](m);
    //初始化迭代次数
    var iter: Int = 0;
    //首先进入外循环，满足最大迭代次数后退出循环
    while (iter < maxIter) {
      //初始化a向量变化统计变量
      var alphaPairsChanged = 0;
      //进入内循环，对a向量中的每一个ai进行优化
      for (i <- 0 until (m)) {
        //选择第i个a
        breakable {
          //初始化ai的取值范围
          var H = 0.0;
          var L = 0.0;
          //计算预测值
          val fxi = (alphas *:* labelMatrix).t * (dataMatrix * dataMatrix(i, ::).t) + b;
          //计算偏差值
          val ei = fxi - labelMatrix(i);
          //如果偏差值不在允许范围内，那么进行优化
          if (((labelMatrix(i) * ei < -toler) && alphas(i) < C) || ((labelMatrix(i) * ei > toler) && alphas(i) > 0)) {
            //随机选择另外一个aj
            val j = selectJrand(i, m);
            //计算aj的预测值和偏差
            val fxj = (alphas *:* labelMatrix).t * (dataMatrix * dataMatrix(j, ::).t) + b;
            val ej = fxj - labelMatrix(j);
            //保留ai和aj的原始值
            val alphaIold = alphas(i);
            val alphaJold = alphas(j);
            //调整a值的上下限
            if (labelMatrix(i) != labelMatrix(j)) {
              L = Math.max(0, alphas(j) - alphas(i));
              H = Math.min(C, C + alphas(j) - alphas(i));
            } else {
              L = Math.max(0, alphas(j) + alphas(i) - C);
              H = Math.min(C, alphas(j) + alphas(i));
            }
            if (L == H) {
              println("L==H")
              break;
            };
            val eta = (2.0 * dataMatrix(i, ::) * dataMatrix(j, ::).t - dataMatrix(i, ::) * dataMatrix(i, ::).t - dataMatrix(j, ::) * dataMatrix(j, ::).t).valueAt(0);
            if (eta >= 0) {
              println("eta>=0");
              break;
            }
            alphas(j) -= labelMatrix(j) * (ei - ej) / eta;
            alphas(j) = clipAlpha(alphas(j), H, L);

            if (Math.abs(alphas(j) - alphaJold) < 0.00001) {
              println("j not moving enough");
              break;
            }
            alphas(i) += labelMatrix(j) * labelMatrix(i) * (alphaJold - alphas(j));

            val b1 = (b - ei - labelMatrix(i) * (alphas(i) - alphaIold) * dataMatrix(i, ::) * dataMatrix(i, ::).t - labelMatrix(j) * (alphas(j) - alphaJold) * dataMatrix(i, ::) * dataMatrix(j, ::).t).valueAt(0);
            val b2 = (b - ej - labelMatrix(i) * (alphas(i) - alphaIold) * dataMatrix(i, ::) * dataMatrix(j, ::).t - labelMatrix(j) * (alphas(j) - alphaJold) * dataMatrix(j, ::) * dataMatrix(j, ::).t).valueAt(0);
            if (0 < alphas(i) && C > alphas(i)) b = b1;
            else if (0 < alphas(j) && C > alphas(j)) b = b2;
            else b = (b1 + b2) / 2.0;
            alphaPairsChanged += 1;
            println(iter, i, alphaPairsChanged);
          }

        }

      }
      if (alphaPairsChanged == 0) iter += 1;
      else iter = 0;
      println(iter);
    }
    return (b, alphas)
    //    print(dataMatrix);
    //    print(labelMatrix);
  }

  def calcWs(alphas: DenseVector[Double], dataArr: Array[Array[Double]], classLabels: Array[Double]): DenseVector[Double] = {
    val (x, labelMatix) = (new DenseMatrix(dataArr.length, dataArr(0).length, dataArr.transpose.flatten), DenseVector(classLabels));
    val (m, n) = (x.rows, x.cols);
    var w = DenseVector.zeros[Double](n);
    for (i <- 0 until (m)) {
      w += x(i, ::).t * alphas(i) * labelMatix(i);
    }
    println(w)
    w
  }




  class Data(val data: Array[Array[Double]], val label: Array[Double], c: Double, t: Double,k:Kernel) {
    val m = data.length; //数据的行数
    val dataM: DenseMatrix[Double] = new DenseMatrix[Double](m, data(0).length, data.transpose.flatten);
    //数据矩阵
    val labelM: DenseVector[Double] = DenseVector(label);
    //标签矩阵
    val C = c;
    //松弛变量
    val toler = t;
    //容错率
    val alphas = DenseVector.zeros[Double](m);
    //a参数
    var b = 0.0;
    //初始化b
    val eCache = DenseMatrix.zeros[Double](m, 2); //矩阵每行对应的误差值，第一类为是否有效，第二列为实际误差值
    val K =  DenseMatrix.zeros[Double](m,m);
    val kernel  = k;
    for(i<- 0 until(m)) {
        K(::,i) := kernelTrans(dataM,dataM(i,::),kernel);
    }
  }
  def kernelTrans(matrix:DenseMatrix[Double], row:Transpose[DenseVector[Double]], kernel: Kernel): DenseVector[Double] ={
    var m = matrix.rows;
    var K:DenseVector[Double] = DenseVector.zeros(m);
    kernel.option._1 match {
      case "lin" => matrix* row.t;//如果是线性核函数，直接进行内积的计算。
      case "rbf" =>{
        for(j<-0 until(m)){
          val deltaRow :Transpose[DenseVector[Double]]= matrix(j,::)-row;
          K(j) = (deltaRow * deltaRow.t);
        }
        K=exp(K/(-1 *  kernel.option._2 *   kernel.option._2));
      }
    }
    K;
  }
  class Kernel{
    var option = ("rbf",1.3);
  }

  def testRBF(): Unit ={
    val (data, lable) = loadDataSet("data/machinelearninginaction/Ch06/testSetRBF.txt");
    val kernel = new Kernel();
    val (m,n) = (data.length,data(0).length);
    val dataM: DenseMatrix[Double] = new DenseMatrix[Double](m, n, data.transpose.flatten);
    val (b, alphas) = smo(data, lable, 200, 0.001, 100,kernel);//根据训练集计算得到b和alphas
    println(b,alphas)

    val svArrbuff = new ArrayBuffer[Array[Double]]();
    val lArrBuff = new ArrayBuffer[Double]();
    val alBuff = new ArrayBuffer[Double]();
    alphas.foreachPair((k,v)=>{
      if(v>0){
        svArrbuff +=data(k);
        lArrBuff +=lable(k);
        alBuff+=v;
      }
    });
    val svArr = svArrbuff.toArray;
    val svs = new DenseMatrix[Double](svArr.length,n,svArr.transpose.flatten);
    val labelSV  = DenseVector[Double](lArrBuff.toArray);
    val alphasSV  = DenseVector[Double](alBuff.toArray);
    println("支持向量的个数",svArrbuff.length)
    var errorCnt = 0.0;
    //数据矩阵
    val labelM: DenseVector[Double] = DenseVector(lable);
    for(i<- 0 until(data.length)){
      val kernelEval = kernelTrans(svs,dataM(i,::),kernel);
      val p = kernelEval.t * (labelSV *:* alphasSV)+b;
      if(p * labelM(i) <0) errorCnt+=1;
    }
    println("训练集错误率",errorCnt/m);

    val (dataTest, lableTest) = loadDataSet("data/machinelearninginaction/Ch06/testSetRBF2.txt");
    val dataMTest: DenseMatrix[Double] = new DenseMatrix[Double](m, n, dataTest.transpose.flatten);
    val labelMTest: DenseVector[Double] = DenseVector(lableTest);
    var errorCntTest = 0.0;
    for(i<- 0 until(data.length)){
      val kernelEval = kernelTrans(svs,dataMTest(i,::),kernel);
      val p = kernelEval.t * (labelSV *:* alphasSV)+b;
      if(p * labelMTest(i) <0) errorCnt+=1;
    }
    println("测试集错误率",errorCnt/m);
  }

  /**
    * SMO的训练算法
    * @param dataMatIn 输入数据
    * @param classLabels 输出数据
    * @param C 松弛变量
    * @param toler 容错率
    * @param maxIter 最大迭代次数
    * @param kernel 核函数
    * @return 返回b值和a向量
    */
  def smo(dataMatIn: Array[Array[Double]], classLabels: Array[Double], C: Double, toler: Double, maxIter: Int,kernel:Kernel): (Double, DenseVector[Double]) = {
    val data = new Data(dataMatIn, classLabels, C, toler,kernel);
    //初始化数据
    var iter = 0;
    //迭代次数
    var alphaPairsChanged = 0;
    //每次迭代a变化的统计
    var isAllData = true; //当前使用的遍历策略，如果是True代表的是全部遍历，False代表的是在非边界a中的进行扫描
    while (iter < maxIter && alphaPairsChanged>0 || isAllData) {
      alphaPairsChanged = 0;
      if (isAllData) {
        for (i <- 0 until (data.m)){
          alphaPairsChanged += inner(i, data)
          printf("全部遍历遍历：第%d次迭代 样本:%d alpha优化次数:%d \n",iter,i,alphaPairsChanged);
        }
      } else {
        data.alphas.foreachPair((k, v) =>
          if (v > 0 && v < C) {
            alphaPairsChanged += inner(k, data)
            printf("非边界遍历：第%d次迭代 样本：%d alpha优化次数:%d \n",iter,k,alphaPairsChanged);
        })
      }
      iter += 1;
      if (isAllData) isAllData = false;
      else if (alphaPairsChanged == 0) isAllData = true;
      println("迭代次数", iter);
    }
    (data.b, data.alphas)
  }


  def inner(i: Int, data: Data): Int = {
    val ei = calE(data, i);
    var is = 0;
    if ( (data.labelM(i) * ei < -data.toler && data.alphas(i) < data.C) || (data.labelM(i) * ei > data.toler && data.alphas(i) > 0)) {
      val (j, ej) = selectJ(data, i, ei);
      val (aIOld, aJOld) = (data.alphas(i), data.alphas(j));
      var L = 0.0;
      var H = 0.0;
      if (data.labelM(i) != data.labelM(j)) {
        L = Math.max(0, data.alphas(j) - data.alphas(i));
        H = Math.min(data.C, data.C + data.alphas(j) - data.alphas(i));
      } else {
        L = Math.max(0, data.alphas(j) + data.alphas(i) - data.C);
        H = Math.min(data.C, data.alphas(j) + data.alphas(i));
      }
      if (L == H) {
        println("L ==H");
        return is;
      }
      val eta = 2.0 * data.K(i, j) -        data.K(i, i) -        data.K(j, j);
      if (eta >= 0.0) {
        println("eta>=0")
        return is;
      }
      data.alphas(j) -= data.labelM(j) * (ei - ej) / eta;
      data.alphas(j) = clipAlpha(data.alphas(j), H, L);
      updateE(data, j);
      if (Math.abs(data.alphas(j) - aJOld) < 0.00001) {
        println("j的变化过小")
        return is;
      }
      data.alphas(i) += data.labelM(j) * data.labelM(i) * (aJOld - data.alphas(j));
      updateE(data, i);
      val b1 = data.b - ei -
        data.labelM(i) * (data.alphas(i) - aIOld) * data.K(i, i) -
        data.labelM(j) * (data.alphas(j) - aJOld) * data.K(i, j);
      val b2 = data.b - ej -
        data.labelM(i) * (data.alphas(i) - aIOld) * data.K(i, j)  -
        data.labelM(j) * (data.alphas(j) - aJOld) * data.K(j, j);
      if (data.alphas(i) > 0 && data.alphas(i) < data.C) {
        data.b = b1;
      } else if (data.alphas(j) > 0 && data.alphas(j) < data.C) {
        data.b = b2;
      }else {
        data.b = (b1+b2)/2;
      }
      is = 1;
    }
    is;
  }

  /**
    * 计算误差
    *
    * @param data 数据对象
    * @param i    当前数据索引
    * @return 误差值E
    */
  def calE(data: Data, i: Int): Double = {
    (data.alphas *:* data.labelM).t * data.K(::,i ) + data.b - data.labelM(i);
  }

  /**
    * 利用启发算法选择j
    *
    * @param data
    * @param i
    * @param ei
    */
  def selectJ(data: Data, i: Int, ei: Double): (Int, Double) = {
    var (j, maxDE, ej, dE) = (0, -1.0, 0.0, 0.0);
    data.eCache(i,::) := DenseVector[Double](1,ei).t;
    var isRandom: Boolean = true;
    data.eCache(::, 0).foreachPair((k, v) => {
      if (v>0&&k != i) {
        isRandom = false;
        val ek = calE(data, k);
        dE = Math.abs(ek - ei);
        if (dE > maxDE) {
          maxDE = dE;
          ej = ek;
          j = k;
        }
      }
    })
    if (isRandom) {
      j = selectJrand(i, data.m);
      ej = calE(data, i);
    }
    (j, ej)
  }

  /**
    * 更新误差e
    * @param data
    * @param i
    */
  def updateE(data: Data, i: Int): Unit = {
    data.eCache(i,::) := DenseVector(1,calE(data,i)).t;
  }

}
