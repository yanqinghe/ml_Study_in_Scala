import breeze.linalg.{DenseMatrix, DenseVector, argmax, max, sum}

import scala.collection.mutable.ArrayBuffer

/**
  * 隐马尔科夫模型的相关算法
  */
object HMM {

  def main(args: Array[String]): Unit = {
    val A = new DenseMatrix[Double](3,3,Array(0.5,0.3,0.2,0.2,0.5,0.3,0.3,0.2,0.5));
    val B = new DenseMatrix[Double](3,2,Array(0.5,0.4,0.7,0.5,0.6,0.3));
    val P = new DenseVector[Double](Array(0.2,0.4,0.4));
    val O = new DenseVector[Int](Array(0,1,0));
    bruteForce(A,B,P,O)
    forward(A,B,P,O);
    backward(A,B,P,O)
    viterbi(A,B,P,O);
  }

  /**
    * 暴力求解算法计算观测序列的概率
    * @param A 状态转移概率矩阵
    * @param B 观测概率矩阵
    * @param P t=1的隐藏概率分布
    * @param O 观测序列
    * @return
    */
  def bruteForce(A:DenseMatrix[Double],B:DenseMatrix[Double],P:DenseVector[Double],O:DenseVector[Int]): Double ={
    val  N = A.rows;//隐藏状态的数目
    val T = O.length;//时间t
    val array = new ArrayBuffer[Double]()//保存结果
    for(j<-0 until(N)){
      cal(A,B,O,array,1,N,j,P(j)*B(j,O(0)));
    }
    def cal(A:DenseMatrix[Double],B:DenseMatrix[Double],O:DenseVector[Int],res:ArrayBuffer[Double],index:Int,n:Int,i:Int,value:Double):Unit={
      if(index==O.length) res+=value;
      else{
        for(j<- 0 until(n)){
          cal(A,B,O,res,index+1,n,j,value * A(i,j) * B(j,O(index)));
        }
      }
    }
    println("暴力求解结果",array.sum)
    array.sum
  }

  /**
    * 前向算法求解HMM序列的概率
    * @param A 状态转移矩阵
    * @param B 观测概率矩阵
    * @param P 初始状态矩阵
    * @param O 观察序列
    * @return
    */
  def forward(A:DenseMatrix[Double],B:DenseMatrix[Double],P:DenseVector[Double],O:DenseVector[Int]):Double={
    val  N = A.rows;//隐藏状态的数目
    val T = O.length;
    val a = DenseVector.zeros[Double](N);
    for(i<- 0 until(N)){
      a(i)=P(i)*B(i,O(0  ));
    }
    for(t<-1 until(T)){
      val arr = new Array[Double](N);
      for(i<-0 until(N)){
        for(j<- 0 until(N)){
          arr(i)+=a(j)*A(j,i)
        }
        arr(i)*=B(i,O(t));
      }
      for (i<-0 until((N))){
        a(i)=arr(i);
      }
    }
    val s = sum(a);
    println("前向算法结果",s);
    s;
  }

  /**
    * 后向算法求HMM观测序列的概率
    * @param A 状态转移矩阵
    * @param B 观测概率矩阵
    * @param P 初始状态矩阵
    * @param O 观察序列
    * @return
    */
  def backward(A:DenseMatrix[Double],B:DenseMatrix[Double],P:DenseVector[Double],O:DenseVector[Int]):Double={
    val N = A.rows;
    val T = O.length;
    val b = DenseVector.ones[Double](N);
    for(t <- (0 until(T-1)).reverse){
      val arr = new Array[Double](N);
      for (i<-0 until(N)){
        for(j <- 0 until(N)){
          arr(i)+= A(i,j)*B(j,O(t+1))*b(j);
        }
      }
      for(i<-0 until(N)){
        b(i)=arr(i)
      }
    }
    for(i<-0 until(N)){
      b(i)=P(i)*B(i,O(0))*b(i);
    }
    val s = sum(b);
    println("后向算法求解结果",s);
    s
  }


  /**
    * 利用维特比算法求解隐藏序列
    * @param A 状态转移矩阵
    * @param B 观测概率矩阵
    * @param P 初始状态矩阵
    * @param O 观察序列
    * @return
    */
  def viterbi(A:DenseMatrix[Double],B:DenseMatrix[Double],P:DenseVector[Double],O:DenseVector[Int]):DenseVector[Int]={
    val N = A.rows;//隐藏状态的数目
    val T = O.length; //时刻T
    //局部状态，delta
    //表示 时刻t 隐藏状态为i的所有可能序列的概率最大值
    //delta_t(i) = max(P(i_t=i,i_t-1,...,i_0,O_1,..,O_t|lambda))
    //迭代公式
    //delta_t+1(i) = max(P(i_t+1=i,i_t,...,i_0,O_1,..,O_t+1|lambda))
    //             = max(delta_t(j)*a_ji)*b_i(O_t+1)
    var delta = P *:* B(::,O(0));
    //局部状态2,psi
    //表示 时刻t隐藏状态为i的所有概率最大的j值，也就是获取得到i状态的的最可能的上一个状态j
    //psi_t(i) = argmax(delta_t(j)*a_ji)
    var psi = DenseMatrix.zeros[Int](N,T);//初始化局部状态1
    var temp  = DenseVector.zeros[Double](N);//初始化局部状态2
    var itmax = new Array[Int](T);//初始化隐藏序列
    for(t<-1 until(T)){
      //从t=1开始遍历
      for(i <-0 until(N)){
        //对状态i进行遍历，首先计算到达状态i的各种序列的概率
        val row= delta *:* A(::,i);
        //然后取其中概率最大的值，j
        psi(i,t) = argmax(row);
        //计算t时刻隐藏序列为i的最大概率值
        temp(i) = max(row);
      }
      //分别t时刻观测状态为O(t)时对应的概率值，即隐藏状态i*观测概率[i,O(t)];
      delta = temp *:* B(::,O(t));
    }
    //取T时刻，隐藏状态i到观测状态O(t)概率最大值，即认为该隐藏状态为最有可能发生的值
    itmax(T-1) = argmax(delta);
    //然后向前回溯，根据局部状态psi的定义，t+1阶段隐藏状态i对应的t阶段隐藏状态j最大概率即是最有可能是的隐藏状态。
    for(t<- (0 until(T-1)).reverse){
      itmax(t) = psi(itmax(t+1),t+1);
    }
    //得到预测的隐藏状态
    val res = new DenseVector[Int](itmax);
    println(res)
    return res;
  }
}
