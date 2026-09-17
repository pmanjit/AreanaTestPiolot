import json, matplotlib.pyplot as plt, os
# Load both experiments
import matplotlib
matplotlib.use('Agg')

# Experiment 1: comparison_2026-09-17T04-38-01.946089.json
with open("logs/comparison_2026-09-17T04-38-01.946089.json") as f:
    exp1 = json.load(f)

with open("logs/comparison_harder_2026-09-17T05-03-11.898886.json") as f:
    exp2 = json.load(f)

# Plot 1: Training loss curves exp1
fig, axes = plt.subplots(2,2, figsize=(14,10))
# Exp1 loss
ax = axes[0,0]
ax.plot(exp1["transformer"]["train_losses"], label="Transformer-64 train", color="blue", linestyle="--")
ax.plot(exp1["transformer"]["val_losses"], label="Transformer val", color="blue", linewidth=2)
ax.plot(exp1["rogm"]["train_losses"], label="ROGM-64 train", color="red", linestyle="--")
ax.plot(exp1["rogm"]["val_losses"], label="ROGM val", color="red", linewidth=2)
ax.set_title("Exp1: Train/Val Loss (seq_len 32)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# Exp1 accuracy
ax = axes[0,1]
ax.plot(exp1["transformer"]["val_accs"], label="Transformer val acc", color="blue")
ax.plot(exp1["transformer"]["val_long_accs"], label="Transformer long(64) acc", color="blue", linestyle="--")
ax.plot(exp1["rogm"]["val_accs"], label="ROGM val acc", color="red")
ax.plot(exp1["rogm"]["val_long_accs"], label="ROGM long(64) acc", color="red", linestyle="--")
ax.set_title("Exp1: Val Accuracy (short 32 vs long 64)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0,1.05)

# Exp1 per-task bar
ax = axes[1,0]
tasks = ["parity","s3","mqar","parity_long","s3_long","mqar_long"]
trans = [exp1["transformer"]["per_task"][t]["acc"] for t in tasks]
rogm = [exp1["rogm"]["per_task"][t]["acc"] for t in tasks]
x = range(len(tasks))
width=0.35
ax.bar([i-width/2 for i in x], trans, width, label="Transformer", color="blue", alpha=0.7)
ax.bar([i+width/2 for i in x], rogm, width, label="ROGM", color="red", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=20)
ax.set_title("Exp1: Per-task Accuracy (final)")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
# annotate
for i,(t,r) in enumerate(zip(trans, rogm)):
    ax.text(i-width/2, t+0.02, f"{t:.2f}", ha='center', fontsize=8, color="blue")
    ax.text(i+width/2, r+0.02, f"{r:.2f}", ha='center', fontsize=8, color="red")

# Exp2 harder overall lengths
ax = axes[1,1]
labels = ["overall_48","overall_96","overall_128","sel_parity_48","sel_parity_128","parity_48","parity_128","s3_48","s3_128"]
t_vals = [exp2["transformer_64"]["metrics"][k]["acc"] for k in labels]
r_vals = [exp2["rogm_64"]["metrics"][k]["acc"] for k in labels]
lite_vals = [exp2["rogm_lite_32"]["metrics"][k]["acc"] for k in labels]
x = range(len(labels))
width=0.25
ax.bar([i-width for i in x], t_vals, width, label="Transformer-64 (77k)", color="blue", alpha=0.7)
ax.bar([i for i in x], r_vals, width, label="ROGM-64 (101k)", color="red", alpha=0.7)
ax.bar([i+width for i in x], lite_vals, width, label="ROGM-lite-32 (26k)", color="green", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([l.replace("_"," ") for l in labels], rotation=30, ha='right', fontsize=8)
ax.set_title("Exp2 Harder: Train 48 → Eval 48/96/128 (selective parity)")
ax.set_ylabel("Accuracy")
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0,1.15)

plt.tight_layout()
plt.savefig("logs/experiment_comparison.png", dpi=150, bbox_inches='tight')
print("Saved logs/experiment_comparison.png")

# Generate second plot: loss curves harder
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(exp2["transformer_64"]["train_losses"], label="Transformer-64 train", color="blue", linestyle="--")
# no val losses stored in harder, only train, but we can approximate
ax.plot(exp2["rogm_64"]["train_losses"], label="ROGM-64 train", color="red", linestyle="--")
ax.plot(exp2["rogm_lite_32"]["train_losses"], label="ROGM-lite-32 train", color="green", linestyle="--")
ax.set_title("Exp2 Harder: Train Loss (seq_len 48, selective parity + S3 + MQAR)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("logs/harder_train_loss.png", dpi=150, bbox_inches='tight')
print("Saved logs/harder_train_loss.png")

# Print summary stats
print("Exp1 ROGM wins:", exp1["_summary"])
print("Exp2 summary:", exp2["_summary"])
