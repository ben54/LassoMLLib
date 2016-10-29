import org.apache.spark.mllib.regression.LassoModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LassoWithSGD
import java.io.File
import java.io.BufferedWriter
import java.io.FileWriter

object Main {
  def main(args: Array[String]) {
    
    val conf = new SparkConf().setAppName("LassoMLLib")
    val sc = new SparkContext(conf)
    
    // Load and parse the data
//    val data = sc.textFile("data/mllib/ridge-data/lpsa.data")
//    val parsedData = data.map { line =>
//      val parts = line.split(',')
//      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
//    }.cache()
    val data = sc.textFile("data/wine.txt")
    val parsedData = data.map { line =>
    val parts = line.split(';')
      LabeledPoint(parts.last.toDouble, Vectors.dense(parts.take(11).map(_.toDouble)))
    }

    
    //To write out the results
    val outputFile = new File("lassoModelDescription.txt")
    val writer = new BufferedWriter(new FileWriter(outputFile))
    
    //going to investigate the effects of various choices of the regularization parameter
    //1.0 makes all weights 0, 0.01 makes all weights non-zero
//    val regularizationParams = Vector(0.001,0.01,0.1,0.5,1.0)
    val regularizationParams = Vector(0.0001,0.001,0.01)
    regularizationParams.foreach { reg =>  
      // Building the model
      val numIterations = 100
      val stepSize = 0.000001
      val miniBatchFraction = 1.0
      val model = LassoWithSGD.train(parsedData, numIterations, reg, miniBatchFraction)
      
//      var regression = new LassoWithSGD().setIntercept(true)
//      regression.optimizer.setStepSize(0.001)
//      val model = regression.run(parsedData)
      
      // Evaluate model on training examples and compute training error
      val valuesAndPreds = parsedData.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }
      val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
      
      //write results to file
      writer.write("Regularization param: " + reg + ", Mean Squared Error: " + MSE + "\n")
      writer.write("Weights: " + model.weights.toArray.mkString(",") + "\n")
//      writer.write(model.toPMML() + "\n\n")
    }
    //close results file writer
    writer.close
    
    // Example of saving and loading model
//    val model = LassoWithSGD.train(parsedData, 10000,0.1, 1.0)
//    model.save(sc, "target/tmp/scalaLassoWithSGDModel")
//    val sameModel = LassoModel.load(sc, "target/tmp/scalaLassoWithSGDModel")
  }
}