import pandas as pd
import numpy as np

# Load timing results
df = pd.read_csv('timing_results.csv')

# Add derived features
df['gates_per_depth'] = df['gate_count'] / (df['logic_depth'] + 1)
df['fanout_variance'] = df['fanout_max'] - df['fanout_avg']
df['complexity_score'] = df['gate_count'] * df['logic_depth'] * df['fanout_avg']

# Calculate target variable (what we want to predict)
# Target: timing slack (negative slack means timing violation)
df['target_slack'] = df['slack_ns']
df['target_delay'] = df['critical_path_delay_ns']

# Feature list
features = [
    'gate_count', 'net_count', 'logic_depth',
    'fanout_max', 'fanout_avg', 'fanout_variance',
    'gates_per_depth', 'complexity_score', 'clock_period_ns'
]

# Save enhanced dataset
df.to_csv('rtl_dataset_features.csv', index=False)

print("Dataset shape:", df.shape)
print("\nFeature summary:")
print(df[features + ['target_slack', 'target_delay']].describe())
print("\nDataset ready for ML training!")
