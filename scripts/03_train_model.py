# scripts/03_train_model.py
import os
from utils import get_spark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, unix_timestamp, to_timestamp, lit

spark = get_spark("ReGenX-Train")

df = spark.table("regenx_clean")

# filter out night rows (no irradiance) and rows missing target/key features
df = df.filter(col("IRRADIATION") > 0).na.drop(subset=["AC_POWER", "IRRADIATION"]) 

# features we will use
feature_cols = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "DC_POWER",
    "hour",
    "day",
    "month",
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

data = assembler.transform(df).select(
    "features",
    col("AC_POWER").alias("label"),
    col("DATE_TIME"),
    col("PLANT_ID"),
    col("SOURCE_KEY")
)

# time-based split to avoid leakage: compute 80th percentile cutoff over time
data = data.withColumn("ts_unix", unix_timestamp(to_timestamp(col("DATE_TIME"))))
cutoff = data.approxQuantile("ts_unix", [0.8], 0.001)[0]
train_df = data.filter(col("ts_unix") <= cutoff).drop("ts_unix")
test_df = data.filter(col("ts_unix") > cutoff).drop("ts_unix")

lr = LinearRegression(featuresCol="features", labelCol="label", regParam=0.1, elasticNetParam=0.5)
lr_model = lr.fit(train_df)
lr_preds = lr_model.transform(test_df).withColumn("model", lit("LinearRegression"))

gbt = GBTRegressor(featuresCol="features", labelCol="label", maxDepth=6, maxIter=60, stepSize=0.1, maxBins=64)
gbt_model = gbt.fit(train_df)
gbt_preds = gbt_model.transform(test_df).withColumn("model", lit("GBTRegressor"))

e_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
e_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

lr_rmse = e_rmse.evaluate(lr_preds)
lr_r2 = e_r2.evaluate(lr_preds)
gbt_rmse = e_rmse.evaluate(gbt_preds)
gbt_r2 = e_r2.evaluate(gbt_preds)

print(f"LinearRegression -> RMSE: {lr_rmse:.4f}  R2: {lr_r2:.4f}")
print(f"GBTRegressor     -> RMSE: {gbt_rmse:.4f}  R2: {gbt_r2:.4f}")

# collect metrics for report export
metrics_lines = [
    f"LinearRegression -> RMSE: {lr_rmse:.4f}  R2: {lr_r2:.4f}",
    f"GBTRegressor     -> RMSE: {gbt_rmse:.4f}  R2: {gbt_r2:.4f}",
]

# save combined predictions with residuals
from pyspark.sql.functions import (col as F_col)  # avoid shadowing col above
combined = lr_preds.unionByName(gbt_preds).select(
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    F_col("label").alias("AC_POWER"),
    F_col("prediction").alias("prediction"),
    (F_col("label") - F_col("prediction")).alias("residual"),
    "model"
)

spark.sql("DROP TABLE IF EXISTS regenx_predictions")
combined.write.mode("overwrite").saveAsTable("regenx_predictions")

print("Saved predictions to regenx_predictions ✅")

# Evaluate GBT on multiple time-based splits (train%, test%)
splits = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.7, 0.3),
]

print("\nGBTRegressor split sensitivity (time-based):")
data_ts = data.withColumn("ts_unix", unix_timestamp(to_timestamp(col("DATE_TIME"))))
for train_ratio, test_ratio in splits:
    q = max(min(train_ratio, 0.99), 0.01)  # keep quantile in (0,1)
    cutoff_q = data_ts.approxQuantile("ts_unix", [q], 0.001)[0]
    tr = data_ts.filter(col("ts_unix") <= cutoff_q).drop("ts_unix")
    te = data_ts.filter(col("ts_unix") > cutoff_q).drop("ts_unix")

    if tr.count() == 0 or te.count() == 0:
        print(f"For {int(train_ratio*100)}%/{int(test_ratio*100)}%: insufficient rows for split")
        continue

    gbt_s = GBTRegressor(featuresCol="features", labelCol="label", maxDepth=6, maxIter=60, stepSize=0.1, maxBins=64)
    gbt_m = gbt_s.fit(tr)
    pr = gbt_m.transform(te)
    rmse_s = e_rmse.evaluate(pr)
    r2_s = e_r2.evaluate(pr)
    line = f"For {int(train_ratio*100)}% train & {int(test_ratio*100)}% test split: RMSE: {rmse_s:.4f}, R2: {r2_s:.4f}"
    print(line)
    metrics_lines.append(f"GBT split {int(train_ratio*100)}-{int(test_ratio*100)}: RMSE: {rmse_s:.4f}, R2: {r2_s:.4f}")

# write metrics and feature importances to output/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

# Model metrics
metrics_path = os.path.join(output_dir, "model_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("\n".join(metrics_lines) + "\n")
print(f"Wrote metrics to {metrics_path}")

# GBT feature importances
fi = gbt_model.featureImportances
importances = [(feature_cols[i], float(fi[i])) for i in range(len(feature_cols))]
fi_df = spark.createDataFrame(importances, ["feature", "importance"]).orderBy(col("importance").desc())
(
    fi_df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(f"file://{output_dir}/gbt_feature_importances")
)
print(f"Wrote feature importances to {output_dir}/gbt_feature_importances/")

spark.stop()
