import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("DecisionTreeMLLib")
    val sc = new SparkContext(conf)
    /*
     * The following came from https://spark.apache.org/docs/1.6.2/mllib-decision-tree.html
     */

    /* 
     * Load and parse the data file. See ./data/mllib/sample_libsvm_data.txt for the data
     * See https://spark.apache.org/docs/latest/mllib-data-types.html for info on formats.
     * The big thing seems to be to put the class as the first value in each row.  This 
     * particular format seems to be for sparse data, where it makes more sense to only list 
     * nonzero values.
     */
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    
    /* Train a DecisionTree model.
     * Empty categoricalFeaturesInfo indicates all features are continuous.
     * For those categorical variables which are continuous, put them into the map
     * as indexOfVariable -> numberOfValuesVariableCanTake.  So, for example, if 
     * the variable with index 5 is weather corresponding to "hot, cool, cold", then
     * put (5,3) into the map and, for that feature, encode _hot_, _cool_, and _cold_
     * as 0,1,2 respectively.
    */
    val categoricalFeaturesInfo = Map[Int, Int]()
    /* Impurity is the way that a decision tree algorithm evaluates a candidate split.  DT's
     * are _greedy_ algorithms - they start at the root and add nodes in a greedy fashion.  They
     * do this because finding the optimal decision tree is NP-complete 
     * (see https://people.csail.mit.edu/rivest/HyafilRivest-ConstructingOptimalBinaryDecisionTreesIsNPComplete.pdf)
     * The greedy algorithm looks at candidate features to split on as it constructs the tree.
     * Candidate splits are evaluated using the impurity measure - those splits which are more pure 
     * (i.e. have low impurity) are preferred.  The impurity is measured by how the training data gets
     * distributed over the nodes on the split.  0 impurity, for instance, would put lead to nodes
     * containing only one class (or, for regression, a single predicted value).  For regression, the only 
     * option MLLib gives you is variance, but that's okay.
     */
    val impurity = "variance"
    /*
     * maxDepth governs how deep the tree can grow.  Deep trees lead to better accuracy but are prone to overfitting 
     * the data.  Controlling the complexity of a model is known as "regularization" and this is one way that 
     * regularization is done in DT algorithms.
     */
    val maxDepth = 5
    //this is another form of regularization that controls how many bins a continuous feature can be split into
    val maxBins = 32
    
    val model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity,
      maxDepth, maxBins)
    
    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    // mean squared error (MSE) is a common way of evaluating the accurcay of a regression model
    val testMSE = labelsAndPredictions.map{ case (v, p) => math.pow(v - p, 2) }.mean()
//    println("Test Mean Squared Error = " + testMSE)
//    println("Learned regression tree model:\n" + model.toDebugString)
    
    // Save and load model
    model.save(sc, "target/tmp/myDecisionTreeRegressionModel")
    val sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeRegressionModel")
    
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression tree model:\n" + model.toDebugString)
    
    
    
  }
}