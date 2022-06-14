import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, sum}
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.DataFrame

object Embeddings {

  def main(args: Array[String]): Unit = {


    //    // Creating Spark objects
    //    val conf = new SparkConf()
    //      .setAppName(appProps.getProperty("spark.app.name"))
    //      .setMaster("local[*]")
    //    val spark = SparkSession.builder()
    //      .appName(appProps.getProperty("spark.app.name"))
    //      .config("spark.streaming.concurrentJobs", 3)
    //      .config(conf)
    //      .getOrCreate()

    val spark: SparkSession = SparkSession
      .builder()
      .master("local[1]")
      .appName(name = "NPL_Spark")
      .config("spark.streaming.concurrentJobs", 3)
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    //Preprocessing
    val data = spark.read.option("header","true")
      .option("sep", ",")
      .option("multiLine", "true")
      .option("quote","\"")
      .option("escape","\"")
      .option("ignoreTrailingWhiteSpace", true)
      .csv("src/main/datasets/jobs.csv")

    var df = data.select("description","fraudulent")
    df.show()
    df.printSchema()

    //Count missing values from dataframe
    println("      Null Values")
    df.select(df.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show
    println("      NaN Values")
    df.select(df.columns.map(c => sum(col(c).isNaN.cast("int")).alias(c)): _*).show
    //val distinctValuesDF = df.select(df("fraudulent")).distinct.show()

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCol("description")
      .setOutputCol("description_t")

    val remover: StopWordsRemover = new StopWordsRemover()
      .setInputCol("description_t")
      .setOutputCol("clean")

    val tokenized: DataFrame = tokenizer.transform(df)
    val filtered: DataFrame = remover.transform(tokenized)
    filtered.show(2000)
  }

}

