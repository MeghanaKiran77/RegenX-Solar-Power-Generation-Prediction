# scripts/05_create_visualizations.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Get paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_root, "output")
visuals_dir = os.path.join(project_root, "visuals")
os.makedirs(visuals_dir, exist_ok=True)

# Load data
print("Loading data...")
import glob

predictions_files = glob.glob(os.path.join(output_dir, "regenx_predictions", "part-*.csv"))
clean_files = glob.glob(os.path.join(output_dir, "regenx_clean", "part-*.csv"))
fi_files = glob.glob(os.path.join(output_dir, "gbt_feature_importances", "part-*.csv"))
metrics_path = os.path.join(output_dir, "model_metrics.txt")

if not predictions_files or not clean_files or not fi_files:
    raise FileNotFoundError("Required CSV files not found. Please run scripts 01-04 first.")

predictions_df = pd.read_csv(predictions_files[0])
clean_df = pd.read_csv(clean_files[0])
fi_df = pd.read_csv(fi_files[0])

# Parse metrics
with open(metrics_path, 'r') as f:
    metrics_lines = f.readlines()

print(f"Loaded {len(predictions_df):,} predictions")
print(f"Loaded {len(clean_df):,} clean data rows")

# Convert DATE_TIME to datetime
predictions_df['DATE_TIME'] = pd.to_datetime(predictions_df['DATE_TIME'])
clean_df['DATE_TIME'] = pd.to_datetime(clean_df['DATE_TIME'])

# Separate predictions by model
lr_preds = predictions_df[predictions_df['model'] == 'LinearRegression']
gbt_preds = predictions_df[predictions_df['model'] == 'GBTRegressor']

# 1. Model Performance Comparison (RMSE and R²)
print("Creating model performance comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Parse metrics for bar chart
models = ['Linear Regression', 'GBT Regressor']
rmse_values = []
r2_values = []

for line in metrics_lines[:2]:
    if 'RMSE:' in line:
        rmse = float(line.split('RMSE:')[1].split()[0])
        r2 = float(line.split('R2:')[1].strip())
        rmse_values.append(rmse)
        r2_values.append(r2)

# RMSE comparison
bars1 = ax1.bar(models, rmse_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax1.set_title('Model Comparison: RMSE', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars1, rmse_values)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

# R² comparison
bars2 = ax2.bar(models, r2_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('Model Comparison: R² Score', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1.1])
ax2.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars2, r2_values)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, '01_model_performance_comparison.png'), bbox_inches='tight')
plt.close()

# 2. Actual vs Predicted Scatter Plots
print("Creating actual vs predicted plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Linear Regression
ax1.scatter(lr_preds['AC_POWER'], lr_preds['prediction'], alpha=0.3, s=10, color='#FF6B6B')
max_val = max(lr_preds['AC_POWER'].max(), lr_preds['prediction'].max())
min_val = min(lr_preds['AC_POWER'].min(), lr_preds['prediction'].min())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual AC_POWER', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted AC_POWER', fontsize=12, fontweight='bold')
ax1.set_title('Linear Regression: Actual vs Predicted', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Calculate R² for display
lr_r2 = 1 - (lr_preds['residual']**2).sum() / ((lr_preds['AC_POWER'] - lr_preds['AC_POWER'].mean())**2).sum()
ax1.text(0.05, 0.95, f'R² = {lr_r2:.4f}', transform=ax1.transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# GBT Regressor
ax2.scatter(gbt_preds['AC_POWER'], gbt_preds['prediction'], alpha=0.3, s=10, color='#4ECDC4')
max_val = max(gbt_preds['AC_POWER'].max(), gbt_preds['prediction'].max())
min_val = min(gbt_preds['AC_POWER'].min(), gbt_preds['prediction'].min())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual AC_POWER', fontsize=12, fontweight='bold')
ax2.set_ylabel('Predicted AC_POWER', fontsize=12, fontweight='bold')
ax2.set_title('GBT Regressor: Actual vs Predicted', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Calculate R² for display
gbt_r2 = 1 - (gbt_preds['residual']**2).sum() / ((gbt_preds['AC_POWER'] - gbt_preds['AC_POWER'].mean())**2).sum()
ax2.text(0.05, 0.95, f'R² = {gbt_r2:.4f}', transform=ax2.transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, '02_actual_vs_predicted.png'), bbox_inches='tight')
plt.close()

# 3. Residual Plots
print("Creating residual plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Linear Regression residuals
ax1.scatter(lr_preds['prediction'], lr_preds['residual'], alpha=0.3, s=10, color='#FF6B6B')
ax1.axhline(y=0, color='r', linestyle='--', lw=2)
ax1.set_xlabel('Predicted AC_POWER', fontsize=12, fontweight='bold')
ax1.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
ax1.set_title('Linear Regression: Residual Plot', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

# GBT Regressor residuals
ax2.scatter(gbt_preds['prediction'], gbt_preds['residual'], alpha=0.3, s=10, color='#4ECDC4')
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted AC_POWER', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
ax2.set_title('GBT Regressor: Residual Plot', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, '03_residual_plots.png'), bbox_inches='tight')
plt.close()

# 4. Feature Importance
print("Creating feature importance plot...")
fig, ax = plt.subplots(figsize=(10, 6))
fi_df_sorted = fi_df.sort_values('importance', ascending=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(fi_df_sorted)))
bars = ax.barh(fi_df_sorted['feature'], fi_df_sorted['importance'], color=colors)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('GBT Regressor: Feature Importance', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, fi_df_sorted['importance'])):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, '04_feature_importance.png'), bbox_inches='tight')
plt.close()

# 5. Time Series of AC_POWER
print("Creating time series plot...")
# Sample data for performance (take every 100th row)
clean_sample = clean_df.iloc[::100].copy()
clean_sample = clean_sample.sort_values('DATE_TIME')

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(clean_sample['DATE_TIME'], clean_sample['AC_POWER'], alpha=0.6, linewidth=0.5, color='#3498db')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('AC_POWER', fontsize=12, fontweight='bold')
ax.set_title('Time Series: AC_POWER Over Time (Sampled)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, '05_time_series.png'), bbox_inches='tight')
plt.close()

# 6. Feature Relationships
print("Creating feature relationship plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# IRRADIATION vs AC_POWER
axes[0, 0].scatter(clean_df['IRRADIATION'], clean_df['AC_POWER'], alpha=0.3, s=5, color='#e74c3c')
axes[0, 0].set_xlabel('IRRADIATION', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('AC_POWER', fontsize=12, fontweight='bold')
axes[0, 0].set_title('IRRADIATION vs AC_POWER', fontsize=13, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# DC_POWER vs AC_POWER
axes[0, 1].scatter(clean_df['DC_POWER'], clean_df['AC_POWER'], alpha=0.3, s=5, color='#2ecc71')
axes[0, 1].set_xlabel('DC_POWER', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('AC_POWER', fontsize=12, fontweight='bold')
axes[0, 1].set_title('DC_POWER vs AC_POWER', fontsize=13, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# MODULE_TEMPERATURE vs AC_POWER
axes[1, 0].scatter(clean_df['MODULE_TEMPERATURE'], clean_df['AC_POWER'], alpha=0.3, s=5, color='#9b59b6')
axes[1, 0].set_xlabel('MODULE_TEMPERATURE', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('AC_POWER', fontsize=12, fontweight='bold')
axes[1, 0].set_title('MODULE_TEMPERATURE vs AC_POWER', fontsize=13, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Hourly average AC_POWER
hourly_avg = clean_df.groupby('hour')['AC_POWER'].mean()
axes[1, 1].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=8, color='#f39c12')
axes[1, 1].set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Average AC_POWER', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Average AC_POWER by Hour of Day', fontsize=13, fontweight='bold')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, '06_feature_relationships.png'), bbox_inches='tight')
plt.close()

# 7. Distribution of AC_POWER
print("Creating distribution plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
ax1.hist(clean_df['AC_POWER'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax1.set_xlabel('AC_POWER', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of AC_POWER', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3, axis='y')

# Box plot by Plant
clean_df.boxplot(column='AC_POWER', by='PLANT_ID', ax=ax2)
ax2.set_xlabel('Plant ID', fontsize=12, fontweight='bold')
ax2.set_ylabel('AC_POWER', fontsize=12, fontweight='bold')
ax2.set_title('AC_POWER Distribution by Plant', fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove default title
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, '07_distributions.png'), bbox_inches='tight')
plt.close()

# 8. GBT Split Sensitivity Analysis
print("Creating split sensitivity plot...")
# Parse split metrics
split_data = []
for line in metrics_lines[2:]:
    line = line.strip()
    if 'GBT split' in line:
        # Format: "GBT split 50-50: RMSE: 23.8476, R2: 0.9958"
        try:
            # Split by colon to separate split info from metrics
            if ':' in line:
                parts = line.split(':', 1)  # Split only on first colon
                split_part = parts[0].strip()  # "GBT split 50-50"
                metrics_part = parts[1].strip()  # "RMSE: 23.8476, R2: 0.9958"
                
                # Extract split percentages
                split_info = split_part.split()[-1]  # "50-50"
                train_pct = int(split_info.split('-')[0])
                
                # Extract RMSE and R2
                if 'RMSE:' in metrics_part and 'R2:' in metrics_part:
                    rmse_str = metrics_part.split('RMSE:')[1].split(',')[0].strip()
                    r2_str = metrics_part.split('R2:')[1].strip()
                    rmse = float(rmse_str)
                    r2 = float(r2_str)
                    split_data.append({'train_pct': train_pct, 'RMSE': rmse, 'R2': r2})
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse line: {line} - {e}")
            continue

if not split_data:
    print("Warning: No split data found. Skipping split sensitivity plot.")
    split_df = None
else:
    split_df = pd.DataFrame(split_data).sort_values('train_pct')

if split_df is not None and len(split_df) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # RMSE by split
    ax1.plot(split_df['train_pct'], split_df['RMSE'], marker='o', linewidth=2, markersize=10, color='#e74c3c')
    ax1.set_xlabel('Training Data Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_title('GBT Model: RMSE Across Different Train-Test Splits', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    for _, row in split_df.iterrows():
        ax1.text(row['train_pct'], row['RMSE'] + 1, f"{row['RMSE']:.2f}", 
                 ha='center', va='bottom', fontweight='bold')

    # R² by split
    ax2.plot(split_df['train_pct'], split_df['R2'], marker='o', linewidth=2, markersize=10, color='#2ecc71')
    ax2.set_xlabel('Training Data Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('GBT Model: R² Score Across Different Train-Test Splits', fontsize=14, fontweight='bold')
    ax2.set_ylim([0.99, 1.0])
    ax2.grid(alpha=0.3)
    for _, row in split_df.iterrows():
        ax2.text(row['train_pct'], row['R2'] + 0.0005, f"{row['R2']:.4f}", 
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, '08_split_sensitivity.png'), bbox_inches='tight')
    plt.close()

print(f"\n✅ All visualizations created successfully!")
print(f"📁 Saved to: {visuals_dir}/")
print("\nGenerated files:")
print("  01_model_performance_comparison.png")
print("  02_actual_vs_predicted.png")
print("  03_residual_plots.png")
print("  04_feature_importance.png")
print("  05_time_series.png")
print("  06_feature_relationships.png")
print("  07_distributions.png")
print("  08_split_sensitivity.png")

