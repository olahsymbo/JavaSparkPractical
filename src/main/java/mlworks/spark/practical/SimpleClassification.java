package mlworks.spark.practical;

import java.io.Serializable;

import org.apache.spark.SparkConf;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

public class TextClassification {

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

        train_df.show(5);

    }

    public class CSVData implements Serializable {
        Double col1;
        Double col2;
        Double col3;
        Double col4;
        String col5;
    }

}
