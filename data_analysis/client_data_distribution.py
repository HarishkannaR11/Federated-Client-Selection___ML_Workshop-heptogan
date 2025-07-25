import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
import torch


#data_path = r"C:\Users\makumarm\Documents\ResearchWork\IISC_workshops\FedratedLearning\ACM_Workshop_SYSML\CIFAR10_dirichlet0.05_12\part_0\CIFAR10_dirichlet0.05_12\train_data.pth"
data_path = r"C:\Users\pinky\Downloads\ACM_Workshop_SYSML-main (1)\ACM_Workshop_SYSML-main\data\CIFAR10_dirichlet0.05_12"

num_clients = 12

summary_stats = []
#Store the label values
#Initialize labels to None
TotalCount = []

for cid in range(num_clients):
    client_dir = os.path.join(data_path, f"part_{cid}", "CIFAR10_dirichlet0.05_12")
    train_file = os.path.join(client_dir, "train_data.pth")
    if not os.path.exists(train_file):
        print(f"Missing: {train_file}")
        continue
    # pickle load
    try:
        with open(train_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded with pickle: {train_file}")
    except Exception as e:
        print(f"pickle.load failed for {train_file}: {e}")
        continue

    print("Type of loaded data:", type(data))
    if isinstance(data, dict):
        print("Keys:", data.keys())
    elif isinstance(data, (list, tuple)):
        print("Length:", len(data))
        TotalCount.append(len(data))
        print("First element type:", type(data[0]))
    elif isinstance(data, torch.Tensor):
        print("Tensor shape:", data.shape)
    else:
        print("Unknown data structure")

    LabelCount = []
    # Print the first 3 tuples
    for i in range(len(data)):
        #Append the label to LabelCount
        LabelCount.append(data[i][1])

    CounterValue = Counter(LabelCount)
    print(f"Counter Value for Client {cid}: {CounterValue}")

    LabelCount = np.array(LabelCount)

    plt.figure(figsize=(6, 3))
    plt.hist(LabelCount, bins=np.arange(13)-0.5, rwidth=0.8, color='skyblue', align='mid', density=True)
    plt.title(f'Client {cid} - Label Distribution (Histogram)')
    plt.xlabel('Class Label')
    plt.ylabel('Density')
    plt.xticks(range(12))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Try to extract labels
    labels = None
    if isinstance(data, dict):
        if 'labels' in data:
            labels = data['labels']
        elif 'targets' in data:
            labels = data['targets']
        else:
            print(f"Could not find 'labels' or 'targets' in {train_file}")
            continue
    elif isinstance(data, (list, tuple)):
        labels = data[1]
    else:
        print(f"Unknown data format in {train_file}")
        continue

    label_count = Counter(labels)

    summary_stats.append({
        "client_id": cid,
        "num_samples": len(labels),
        "unique_classes": len(set(labels)),
        "label_counts": label_count
    })

# Plot the Total Number of the samples per client
plt.figure(figsize=(8, 4))
plt.bar(range(num_clients), TotalCount, color='orange')
plt.xlabel("Client ID")
plt.ylabel("Number of Samples")
plt.title("Number of Samples per Client")
plt.grid(axis='y')
plt.xticks(range(num_clients))
plt.tight_layout()
plt.show()