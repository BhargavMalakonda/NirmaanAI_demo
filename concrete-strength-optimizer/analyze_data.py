import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv(r"E:\hackathons\SEM-4\CreaTech Hackk\concrete-strength-optimizer\dataset.csv")

print("=== DATASET ANALYSIS ===")
print(f"\nShape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Basic Statistics ===")
print(df.describe())

print("\n=== Curing Types ===")
print(df['curing_type'].value_counts())

print("\n=== Strength Range ===")
print(f"Min: {df['strength'].min()} MPa")
print(f"Max: {df['strength'].max()} MPa")
print(f"Mean: {df['strength'].mean():.2f} MPa")

print("\n=== Hours Range ===")
print(f"Min hours: {df['hours'].min()}")
print(f"Max hours: {df['hours'].max()}")
print(f"Mean hours: {df['hours'].mean():.1f}")

# Check correlation
print("\n=== Correlation with Strength ===")
correlations = df[['temperature', 'humidity', 'hours', 'strength']].corr()['strength'].sort_values(ascending=False)
print(correlations)

# Sample some records
print("\n=== Sample Records (First 10) ===")
print(df.head(10))