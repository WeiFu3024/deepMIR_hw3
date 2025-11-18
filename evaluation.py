import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

env = os.environ.copy()
cwd = 'MusDr'
dictinary_path = 'basic_event_dictionary.pkl'

folder_to_evaluate = [
    'results/gpt2_epoch_100',
    'results/gpt2_epoch_150',
    'results/gpt2_epoch_200',
    'results/transformer_xl_epoch_100',
    'results/transformer_xl_epoch_150',
    'results/transformer_xl_epoch_200',
    'results/cpword_50',
    'results/cpword_36',
    'results/cpword_30',
    'results/compose_embellish/stage2'
]

dictinary_path = os.path.abspath(dictinary_path)
for folder in folder_to_evaluate:
    folder = os.path.abspath(folder)
    print(f"Evaluating results in folder: {folder}")
    cmd = [
        'python', '-m', 'musdr.eval_metrics',
        '--dict_path', dictinary_path,
        '--output_file_path', folder,
    ]
    subprocess.run(cmd, cwd=cwd, env=env)
    cp_cmd = [
        'cp',
        os.path.join('pop1k7.csv'),
        os.path.join(folder, 'evaluation_results.csv')
    ]
    subprocess.run(cp_cmd, cwd=cwd, env=env)
    
# Calculate Average Metrics Across All Experiments, then Plot and Save
print("Calculating average metrics across all experiments...")
average_metrics = {}
std_metrics = {}
for folder in folder_to_evaluate:
    csv_path = os.path.join(folder, 'evaluation_results.csv')
    df = pd.read_csv(csv_path)
    # filter out file contain 'skyline' (for compose and embellish results)
    df = df[~df['piece_name'].str.contains('skyline')]
    print(f'Number of pieces evaluated in {folder}: {len(df)}')

    avg_metrics = df.iloc[:, 1:].mean()
    std_metrics_values = df.iloc[:, 1:].std()
    average_metrics[folder] = avg_metrics
    std_metrics[folder] = std_metrics_values
    print(f"Average metrics for {folder}:")
    for (metric_avg, value_avg), (metric_std, value_std) in zip(avg_metrics.items(), std_metrics_values.items()):
        assert metric_avg == metric_std, "Metric names do not match!"
        print(f"  {metric_avg}: {value_avg:.4f} ± {value_std:.4f}")
    print('=' * 50)

# Convert average metrics to DataFrame for easier plotting
avg_metrics_df = pd.DataFrame(average_metrics).T
avg_metrics_df.to_csv('results/average_evaluation_metrics.csv')
std_metrics_df = pd.DataFrame(std_metrics).T
std_metrics_df.to_csv('results/std_evaluation_metrics.csv')

# Plot average metrics as grouped bar chart with error bars
plt.figure(figsize=(14, 8))
num_experiments = len(avg_metrics_df)
num_metrics = len(avg_metrics_df.columns)
x = np.arange(num_experiments)
total_width = 0.8
bar_width = total_width / max(1, num_metrics)

# Plot each metric separately as its own bar chart with error bars
os.makedirs('results', exist_ok=True)

for metric in avg_metrics_df.columns:
    vals = avg_metrics_df[metric].values
    errs = std_metrics_df[metric].values

    # compute a small offset for annotation based on data range for this metric
    y_max = (vals + errs).max()
    label_offset = y_max * 0.01 if y_max > 0 else 0.001

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.6
    bars = ax.bar(x, vals, bar_width, yerr=errs, capsize=5, color='C0')

    # add vertical value on top of each bar
    for xi, v in zip(x, vals):
        ax.text(xi, v + label_offset, f"{v:.4f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(avg_metrics_df.index, rotation=45, ha='right')
    ax.set_xlabel('Experiment Folder')
    ax.set_ylabel(metric)
    ax.set_title(f'Average {metric} Across Experiments')
    ax.set_ylim(bottom=0 if (vals - errs).min() >= 0 else None)
    plt.tight_layout()

    safe_metric = metric.replace(' ', '_').replace('/', '_')
    out_path = f'results/average_metric_{safe_metric}.png'
    plt.savefig(out_path)
    plt.close(fig)

    print(f"Saved plot for metric '{metric}' to '{out_path}'")

print("Evaluation and per-metric plotting completed.")