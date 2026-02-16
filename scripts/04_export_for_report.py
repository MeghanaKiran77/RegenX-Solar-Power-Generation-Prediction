# scripts/04_export_for_report.py
import os
from utils import get_spark


def export_table(spark, table_name: str, output_dir: str, file_stem: str) -> None:
    df = spark.table(table_name)
    (
        df.coalesce(1)
          .write.mode("overwrite")
          .option("header", True)
          .csv(f"file://{output_dir}/{file_stem}")
    )


if __name__ == "__main__":
    spark = get_spark("ReGenX-Export")

    # Resolve project root and output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Export required tables
    export_table(spark, "regenx_raw", output_dir, "regenx_raw")
    export_table(spark, "regenx_clean", output_dir, "regenx_clean")
    export_table(spark, "regenx_predictions", output_dir, "regenx_predictions")

    print("Exported CSVs to output/:")
    print("- output/regenx_raw/")
    print("- output/regenx_clean/")
    print("- output/regenx_predictions/")

    spark.stop()


