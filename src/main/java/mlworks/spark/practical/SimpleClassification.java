package mlworks.spark.practical;

import java.io.Serializable;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

public class SimpleClassification {

    public static void main(String[] args){

        //Create a SparkContext to initialize
        SparkConf conf = new SparkConf().setMaster("local").setAppName("Iris-Flower Classification");
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaML")
                .config("spark.master", "local")
                .getOrCreate();

        StructType schema = new StructType()
                .add("sepal length", "double")
                .add("sepal width", "double")
                .add("petal length", "double")
                .add("petal width", "double")
                .add("species", "string");

        Dataset<Row> train_df = spark.read()
                .option("mode", "DROPMALFORMED")
                .schema(schema)
                .csv("/Users/o.arigbabu/IdeaProjects/JavaSparkPractical/data/train.csv");

        Dataset<Row> test_df = spark.read()
                .option("mode", "DROPMALFORMED")
                .schema(schema)
                .csv("/Users/o.arigbabu/IdeaProjects/JavaSparkPractical/data/test.csv");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"sepal length", "sepal width", "petal length", "petal width"})
                .setOutputCol("features");

        StringIndexer indexer = new StringIndexer()
                .setInputCol("species")
                .setOutputCol("transformedSpecies");

        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("transformedSpecies")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] { indexer, assembler, rf });

        PipelineModel model = pipeline.fit(train_df);

        Dataset<Row> predictions = model.transform(test_df);

        System.out.println("Test prediction results");
        predictions.show();

        // Get evaluation metrics.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("transformedSpecies")
                .setPredictionCol("prediction")
                .setMetricName("precision");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));
    }

}
