import matplotlib.pyplot as plt
import json

# Data from ablation logs (22-epoch full where available, quick 15-epoch for others but normalized)
# Use metrics directly from logs we captured

# Let's manually define dict for plotting based on above summary
data = {
    "Transformer": {"params":77, "overall_48":0.48, "overall_128":0.421, "parity_128":0.540, "s3_128":0.189, "sel_128":0.539},
    "ROGM-full\n101k": {"params":101, "overall_48":0.984, "overall_128":0.737, "parity_128":0.785, "s3_128":0.693, "sel_128":0.777},
    "singleH\n93k": {"params":93, "overall_48":0.816, "overall_128":0.597, "parity_128":0.725, "s3_128":0.331, "sel_128":0.752},
    "noOsc\n93k": {"params":93, "overall_48":0.774, "overall_128":0.768, "parity_128":0.992, "s3_128":0.339, "sel_128":0.998},
    "noMem\n85k": {"params":85, "overall_48":0.766, "overall_128":0.612, "parity_128":0.780, "s3_128":0.285, "sel_128":0.784},
    "minimal\n85k": {"params":85, "overall_48":0.753, "overall_128":0.730, "parity_128":0.999, "s3_128":0.211, "sel_128":0.999},
    "lite-32\n26k": {"params":26, "overall_48":0.985, "overall_128":0.764, "parity_128":0.808, "s3_128":0.68, "sel_128":0.818},
}

labels = list(data.keys())
metrics = ["overall_48","overall_128","parity_128","s3_128","sel_128"]
colors = {"overall_48":"#4e79a7","overall_128":"#59a14f","parity_128":"#f28e2b","s3_128":"#e15759","sel_128":"#76b7b2"}

fig, axes = plt.subplots(2,1, figsize=(14,10))

# Top: per-variant accuracy bars grouped by metric
import numpy as np
x = np.arange(len(labels))
width = 0.15
for i, m in enumerate(metrics):
    vals = [data[l][m] for l in labels]
    axes[0].bar(x + (i-2)*width, vals, width, label=m.replace('_',' '), color=colors[m], alpha=0.9)
    for j, v in enumerate(vals):
        axes[0].text(x[j] + (i-2)*width, v+0.02, f"{v:.2f}", ha='center', fontsize=7, rotation=90)

axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=9)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Ablation: Which ROGM Component is Essential? (train 48 → eval 48/128 harder)")
axes[0].legend(ncol=3, fontsize=8)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim(0,1.15)
axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance parity')
axes[0].axhline(0.166, color='gray', linestyle=':', alpha=0.5, label='chance S3')

# Bottom: delta vs full for S3 (most discriminative)
full_s3_128 = data["ROGM-full\n101k"]["s3_128"]
deltas = {k: v["s3_128"] - full_s3_128 for k,v in data.items() if k!="ROGM-full\n101k"}
# also parity delta
full_parity = data["ROGM-full\n101k"]["parity_128"]
deltas_parity = {k: v["parity_128"] - full_parity for k,v in data.items() if k!="ROGM-full\n101k"}

ax = axes[1]
variants = list(deltas.keys())
s3_deltas = [deltas[v] for v in variants]
parity_deltas = [deltas_parity[v] for v in variants]
x2 = np.arange(len(variants))
ax.bar(x2 - 0.2, s3_deltas, 0.4, label="S3 128 delta vs full", color="#e15759")
ax.bar(x2 + 0.2, parity_deltas, 0.4, label="Parity 128 delta vs full", color="#f28e2b")
for i, (s,p) in enumerate(zip(s3_deltas, parity_deltas)):
    ax.text(x2[i]-0.2, s - 0.05 if s<0 else s+0.02, f"{s:+.2f}", ha='center', fontsize=8, color="#e15759", fontweight='bold')
    ax.text(x2[i]+0.2, p - 0.05 if p<0 else p+0.02, f"{p:+.2f}", ha='center', fontsize=8, color="#f28e2b")
ax.set_xticks(x2)
ax.set_xticklabels(variants, fontsize=9)
ax.set_ylabel("Accuracy Delta vs Full")
ax.set_title("Ablation Impact: Dropping Second Householder or Osc or Memory Cripples S3 (-0.35 to -0.48) but Parity Survives")
ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(-0.10, color='red', linestyle='--', alpha=0.5, label='essential threshold -0.10')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(-0.6, 0.4)

plt.tight_layout()
plt.savefig("logs/ablation_analysis.png", dpi=150, bbox_inches='tight')
print("Saved logs/ablation_analysis.png")

# Also generate efficiency plot: acc per param
fig, ax = plt.subplots(figsize=(8,6))
for label, d in data.items():
    # efficiency = s3_128 / params *100? just scatter
    ax.scatter(d["params"], d["s3_128"], s=100, label=label)
    ax.annotate(label.replace("\n"," "), (d["params"], d["s3_128"]), textcoords="offset points", xytext=(5,5), fontsize=8)
ax.set_xlabel("Params (k)")
ax.set_ylabel("S3 Accuracy at 128 (hard)")
ax.set_title("Parameter Efficiency: lite-32 (26k) achieves 0.68 S3 at 128 vs Transformer 0.189 at 77k")
ax.grid(True, alpha=0.3)
# add transformer baseline line
ax.axhline(0.189, color='blue', linestyle='--', alpha=0.5, label='Transformer S3 0.189')
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("logs/efficiency.png", dpi=150, bbox_inches='tight')
print("Saved logs/efficiency.png")
