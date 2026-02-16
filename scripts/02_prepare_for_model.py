# scripts/02_prepare_for_model.py
from utils import get_spark
from pyspark.sql.functions import col

spark = get_spark("ReGenX-Prep")

# load the table you just created
df = spark.table("regenx_raw")

# we want to predict AC_POWER (grid side)
# drop rows with missing AC_POWER or IRRADIATION (super important feature)
df_clean = df.na.drop(subset=["AC_POWER", "IRRADIATION"])

# pick only columns we care about
cols_to_keep = [
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    "AC_POWER",
    "DC_POWER",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "hour",
    "day",
    "month"
]

df_clean = df_clean.select(*cols_to_keep)

# overwrite / create the clean table
spark.sql("DROP TABLE IF EXISTS regenx_clean")
df_clean.write.mode("overwrite").saveAsTable("regenx_clean")

print("Prepared regenx_clean ✅")
print(f"Row count: {df_clean.count()}")

spark.stop()
