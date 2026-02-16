# ReGenX: Solar Power Generation Prediction Project

## Executive Summary

This project implements a machine learning pipeline to predict AC power output from solar photovoltaic plants using Apache Spark. The system processes data from two solar plants, performs feature engineering, and trains regression models to forecast power generation based on weather conditions and temporal features.

## 1. Project Objectives

- Build a scalable data processing pipeline using Apache Spark
- Predict AC power output from solar plants using weather and generation data
- Compare baseline linear regression with advanced gradient-boosted tree models
- Evaluate model performance across different train-test split ratios
- Export processed data and model results for analysis and reporting

## 2. Data Sources

The project uses four CSV datasets from Kaggle (check "data" folder for more details):
- **Plant 1 Generation Data**: Inverter-level power generation data (68,780 rows)
- **Plant 1 Weather Sensor Data**: Weather measurements (3,184 rows)
- **Plant 2 Generation Data**: Inverter-level power generation data (67,700 rows)
- **Plant 2 Weather Sensor Data**: Weather measurements (3,261 rows)

**Key Variables:**
- **Target**: AC_POWER (grid-side AC power output)
- **Features**: DC_POWER, IRRADIATION, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE
- **Identifiers**: PLANT_ID, SOURCE_KEY (inverter ID), DATE_TIME
- **Temporal**: hour, day, month (extracted from timestamps)

## 3. Data Pipeline Architecture

### 3.1 Infrastructure Setup

**Technology Stack:**
- Apache Spark 4.0.1 (distributed processing)
- PySpark (Python API)
- Hive Metastore (table management)
- Local filesystem storage

**Key Components:**
- `utils.py`: SparkSession configuration with Hive support
- Local mode execution with warehouse directory: `/tmp/spark-warehouse`

### 3.2 Data Processing Pipeline

#### Stage 1: Data Ingestion and Integration (`01_build_dataset.py`)

**Challenges Addressed:**
1. **Date Format Inconsistency**: Plant 1 generation data uses `dd-MM-yyyy HH:mm` format, while other files use `yyyy-MM-dd HH:mm:ss`
2. **Data Integration**: Multiple inverters per timestamp require careful joining strategy

**Process:**
1. Read all four CSV files from local filesystem using `file://` protocol
2. Parse timestamps with format-specific handling:
   - Plant 1 generation: `dd-MM-yyyy HH:mm`
   - All other files: `yyyy-MM-dd HH:mm:ss`
3. Extract temporal features: hour, day, month
4. Normalize DATE_TIME to consistent string format for joining
5. Join generation and weather data on DATE_TIME and PLANT_ID (left join)
6. Union Plant 1 and Plant 2 datasets
7. Save as managed Hive table: `regenx_raw`

**Output:**
- Table: `regenx_raw`
- Row count: 136,476 rows
- Location: `/tmp/spark-warehouse/regenx_raw/`

#### Stage 2: Data Cleaning and Feature Selection (`02_prepare_for_model.py`)

**Process:**
1. Load `regenx_raw` table
2. Remove rows with missing AC_POWER or IRRADIATION (critical features)
3. Select relevant columns for modeling:
   - Identifiers: DATE_TIME, PLANT_ID, SOURCE_KEY
   - Target: AC_POWER
   - Features: DC_POWER, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION
   - Temporal: hour, day, month
4. Save as managed Hive table: `regenx_clean`

**Output:**
- Table: `regenx_clean`
- Row count: 136,472 rows (4 rows removed due to missing values)

#### Stage 3: Model Training and Evaluation (`03_train_model.py`)

**Feature Engineering:**
- Assembled features: IRRADIATION, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, DC_POWER, hour, day, month
- Filtered out night rows (IRRADIATION > 0) for training

**Models Trained:**

1. **Linear Regression (Baseline)**
   - Regularization: Elastic Net (L1 + L2)
   - Parameters: regParam=0.1, elasticNetParam=0.5
   - Purpose: Baseline comparison

2. **Gradient-Boosted Trees Regressor (GBT)**
   - Parameters: maxDepth=6, maxIter=60, stepSize=0.1, maxBins=64
   - Purpose: Capture non-linear relationships and feature interactions

**Evaluation Strategy:**
- **Time-based splitting**: Prevents data leakage by splitting on temporal order
- Baseline: 80% train / 20% test (earliest 80% timestamps for training)
- Sensitivity analysis: Multiple train-test ratios (50-50, 60-40, 40-60, 30-70, 70-30)

**Metrics:**
- RMSE (Root Mean Squared Error): Lower is better
- R² (Coefficient of Determination): Higher is better (0-1 scale)

**Results:**

| Model | Train-Test Split | RMSE | R² |
|-------|-----------------|------|-----|
| Linear Regression | 80-20 | 160.01 | 0.7728 |
| GBT Regressor | 80-20 | 16.63 | 0.9975 |
| GBT Regressor | 70-30 | 18.80 | 0.9972 |
| GBT Regressor | 60-40 | 21.27 | 0.9966 |
| GBT Regressor | 50-50 | 23.85 | 0.9958 |
| GBT Regressor | 40-60 | 33.00 | 0.9922 |
| GBT Regressor | 30-70 | 28.36 | 0.9944 |

**Key Findings:**
1. GBT significantly outperforms Linear Regression (RMSE: 16.63 vs 160.01)
2. Model performance is stable across different split ratios (R² > 0.99)
3. Performance degrades slightly with less training data (30-70 split: R² = 0.9944)
4. No signs of overfitting: consistent performance across splits

**Feature Importances (GBT Model):**
- Feature importance analysis exported to `output/gbt_feature_importances/`
- Most important features: IRRADIATION, DC_POWER, MODULE_TEMPERATURE

**Outputs:**
- Predictions table: `regenx_predictions` (includes both LR and GBT predictions with residuals)
- Model metrics: `output/model_metrics.txt`
- Feature importances: `output/gbt_feature_importances/`

#### Stage 4: Data Export (`04_export_for_report.py`)

**Exported Files:**
1. `output/regenx_raw/` - Raw integrated dataset (CSV)
2. `output/regenx_clean/` - Cleaned dataset for modeling (CSV)
3. `output/regenx_predictions/` - Model predictions with residuals (CSV)
4. `output/model_metrics.txt` - Performance metrics for all models
5. `output/gbt_feature_importances/` - Feature importance rankings (CSV)

## 4. Technical Implementation Details

### 4.1 Date Format Handling

The project handles inconsistent date formats by:
1. Detecting format during parsing
2. Converting to timestamp using format-specific patterns
3. Normalizing to standard string format for consistent joining
4. Extracting temporal features (hour, day, month) for modeling

### 4.2 Time-Based Data Splitting

To prevent temporal data leakage:
- Data sorted by timestamp
- Training set: earliest N% of timestamps
- Test set: latest (100-N)% of timestamps
- Ensures model doesn't see future data during training

### 4.3 Model Selection Rationale

**Linear Regression:**
- Fast training and inference
- Interpretable coefficients
- Baseline for comparison
- Limitations: Cannot capture non-linear relationships

**Gradient-Boosted Trees:**
- Handles non-linear relationships
- Captures feature interactions automatically
- Robust to outliers
- Excellent performance on structured data
- Trade-off: Less interpretable, longer training time

## 5. Results and Analysis

### 5.1 Model Performance

The GBT model achieves exceptional performance:
- **R² = 0.9975**: Explains 99.75% of variance in AC power
- **RMSE = 16.63**: Average prediction error of ~17 units
- **Stability**: Consistent performance across different train-test splits

### 5.2 Overfitting Assessment

The model shows no signs of overfitting:
- Performance remains high (R² > 0.99) across all split ratios
- Slight degradation with less training data is expected and minimal
- Model generalizes well to unseen temporal data

### 5.3 Feature Importance

Key drivers of AC power prediction:
1. **IRRADIATION**: Primary driver (solar energy input)
2. **DC_POWER**: Direct current output from panels
3. **MODULE_TEMPERATURE**: Affects panel efficiency
4. **AMBIENT_TEMPERATURE**: Environmental context
5. **Temporal features**: Capture daily/seasonal patterns

## 6. Project Deliverables

### 6.1 Code Repository

**Scripts:**
- `scripts/utils.py`: SparkSession utility
- `scripts/01_build_dataset.py`: Data ingestion and integration
- `scripts/02_prepare_for_model.py`: Data cleaning and feature selection
- `scripts/03_train_model.py`: Model training and evaluation
- `scripts/04_export_for_report.py`: Data export utility

**Configuration:**
- `requirements.txt`: Python dependencies (pyspark==4.0.1, py4j==0.10.9.9)
- `.vscode/settings.json`: IDE configuration
- `pyrightconfig.json`: Type checking configuration

### 6.2 Data Artifacts

**Tables (Hive Metastore):**
- `regenx_raw`: Raw integrated dataset
- `regenx_clean`: Cleaned dataset
- `regenx_predictions`: Model predictions

**Exported Files:**
- CSV files in `output/` directory
- Model metrics and feature importances

### 6.3 Documentation

- This project report
- Inline code comments
- README.md (project overview)

## 7. Challenges and Solutions

### Challenge 1: Date Format Inconsistency
**Solution**: Implemented format-specific parsing with normalization to standard format

### Challenge 2: HDFS vs Local Filesystem
**Solution**: Configured Spark to use local filesystem with `file://` protocol and local mode

### Challenge 3: Temporal Data Leakage
**Solution**: Implemented time-based splitting instead of random splitting

### Challenge 4: Missing Dependencies
**Solution**: Created requirements.txt and installed py4j dependency

## 8. Future Enhancements

1. **Hyperparameter Tuning**: Grid search or random search for optimal GBT parameters
2. **Cross-Validation**: K-fold cross-validation for more robust evaluation
3. **Feature Engineering**: Additional features (day of week, season, weather interactions)
4. **Model Interpretability**: SHAP values for feature importance explanation
5. **Real-time Prediction**: Streaming pipeline for real-time power prediction
6. **Per-Inverter Models**: Train separate models for each inverter for improved accuracy
7. **Time Series Models**: LSTM or ARIMA for temporal pattern capture
8. **Deployment**: Model serving API for production use

## 9. Conclusion

This project successfully demonstrates:
- Scalable data processing with Apache Spark
- Effective handling of data quality issues (format inconsistencies, missing values)
- Strong model performance (R² = 0.9975) using gradient-boosted trees
- Robust evaluation methodology with time-based splitting
- Complete end-to-end pipeline from raw data to predictions

The GBT model significantly outperforms linear regression and shows excellent generalization across different data splits, making it suitable for production deployment in solar power forecasting applications.

---

**Project Statistics:**
- Total rows processed: 136,476
- Models trained: 2 (Linear Regression, GBT)
- Evaluation splits: 6 (1 baseline + 5 sensitivity)
- Final model performance: RMSE = 16.63, R² = 0.9975

