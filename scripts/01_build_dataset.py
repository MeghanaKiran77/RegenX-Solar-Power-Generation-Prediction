from utils import get_spark
from pyspark.sql.functions import to_timestamp, hour, dayofmonth, month
import os

# Get the project root directory (parent of scripts/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spark = get_spark("ReGenX-Build")

# 1) read plant 1
p1_gen = spark.read.csv(f"file://{project_root}/data/Plant_1_Generation_Data.csv", header=True, inferSchema=True)
p1_wth = spark.read.csv(f"file://{project_root}/data/Plant_1_Weather_Sensor_Data.csv", header=True, inferSchema=True)

# 2) read plant 2
p2_gen = spark.read.csv(f"file://{project_root}/data/Plant_2_Generation_Data.csv", header=True, inferSchema=True)
p2_wth = spark.read.csv(f"file://{project_root}/data/Plant_2_Weather_Sensor_Data.csv", header=True, inferSchema=True)

# DATE_TIME formats vary: Plant 1 gen uses "dd-MM-yyyy HH:mm", others use "yyyy-MM-dd HH:mm:ss"
def add_time_cols(df, date_format="yyyy-MM-dd HH:mm:ss"):
    """Convert DATE_TIME string to timestamp and extract time components."""
    df = df.withColumn("ts", to_timestamp(df.DATE_TIME, date_format))
    # Convert timestamp back to standard string format for consistent joining
    df = df.withColumn("DATE_TIME", df.ts.cast("string"))
    df = df.withColumn("hour", hour(df.ts)) \
           .withColumn("day", dayofmonth(df.ts)) \
           .withColumn("month", month(df.ts))
    return df

# Plant 1 generation data uses different format: "dd-MM-yyyy HH:mm"
p1_gen = add_time_cols(p1_gen, "dd-MM-yyyy HH:mm")
# All other files use standard format: "yyyy-MM-dd HH:mm:ss"
p1_wth = add_time_cols(p1_wth)
p2_gen = add_time_cols(p2_gen)
p2_wth = add_time_cols(p2_wth)

# join logic:
# generation: many inverters (SOURCE_KEY = inverter id)
# weather: one sensor row per timestamp per plant
# so: join on DATE_TIME and PLANT_ID
p1_joined = p1_gen.join(
    p1_wth.select("DATE_TIME", "PLANT_ID", "AMBIENT_TEMPERATURE",
                  "MODULE_TEMPERATURE", "IRRADIATION"),
    on=["DATE_TIME", "PLANT_ID"],
    how="left"
)

p2_joined = p2_gen.join(
    p2_wth.select("DATE_TIME", "PLANT_ID", "AMBIENT_TEMPERATURE",
                  "MODULE_TEMPERATURE", "IRRADIATION"),
    on=["DATE_TIME", "PLANT_ID"],
    how="left"
)

# union both plants to make a bigger dataset
all_data = p1_joined.unionByName(p2_joined)

# drop table if it exists, then save as a managed table
spark.sql("DROP TABLE IF EXISTS regenx_raw")
all_data.write.mode("overwrite").saveAsTable("regenx_raw")

spark.stop()
