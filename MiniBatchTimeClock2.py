import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob

# Find all result files and extract batch sizes without using re
result_files = sorted(glob.glob("miniBatch results/results_batchsize_*.pkl"),
                      key=lambda x: int(x.split("_")[-1].split(".")[0]))
batch_sizes = [int(fname.split("_")[-1].split(".")[0]) for fname in result_files]

all_minibatch_times = []

for fname in result_files:
    with open(fname, "rb") as f:
        data = pickle.load(f)
    minibatch_times = data['minibatch_times']
    minibatch_times = np.array(minibatch_times)
    if minibatch_times.ndim == 2:
        minibatch_times = minibatch_times.flatten()
    all_minibatch_times.append(minibatch_times)

plt.figure(figsize=(12, 6))
box = plt.boxplot(all_minibatch_times, labels=[str(bs) for bs in batch_sizes], medianprops=dict(color='red', linewidth=2))
plt.xlabel('Batch Size')
plt.ylabel('Minibatch Time (seconds)')
plt.title('Minibatch Time Distribution per Batch Size')
plt.grid()
plt.tight_layout()

# Annotate median values
for i, median_line in enumerate(box['medians']):
    x, y = median_line.get_xydata()[1]
    plt.text(x, y, f'{y:.4f}', color='red', ha='center', va='bottom', fontweight='bold')

plt.show()