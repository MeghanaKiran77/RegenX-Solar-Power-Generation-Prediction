from pyspark.sql import SparkSession 


def get_spark(app_name: str) -> SparkSession:
    """
    Create or get a SparkSession with Hive support enabled.
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        SparkSession instance with Hive support
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse") \
        .enableHiveSupport() \
        .getOrCreate()
    
    return spark

