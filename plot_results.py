import matplotlib.pyplot as plt
import numpy as np

data = [
    ("CNN + XGBoost",        0.5633),
    ("ResNet18 (run 1)",     0.6326),
    ("ResNet34 (run 1)",     0.6342),
    ("ResNet18 (run 3)",     0.6447),
    ("ResNet34 (run 2)",     0.6496),
    ("ResNet18 (run 2)",     0.6509),
    ("Simple ensemble",      0.6542),
    ("Stacked ensemble",     0.6564),
]
names = [d[0] for d in data]
scores = [d[1] for d in data]
n = len(data)

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)

cmap = plt.cm.Blues
colors = [cmap(0.35 + 0.55 * (i / (n - 1))) for i in range(n)]
colors[-1] = "#f59e0b"

y = np.arange(n)
bars = ax.barh(y, scores, color=colors, height=0.62, edgecolor="white", linewidth=0.6)

ax.axvline(0.50, color="red", linestyle="--", linewidth=1.4)
ax.text(0.50, n - 0.3, " random\n guess", color="red", fontsize=9.5, va="top")

for i, sc in enumerate(scores):
    weight = "bold" if i == n - 1 else "normal"
    ax.text(sc + 0.006, i, f"{sc:.4f}", va="center", fontsize=11, fontweight=weight)

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=11.5)
ax.set_xlim(0.40, 0.72)
ax.set_xlabel("Balanced Accuracy (OOF)")
ax.set_title("Lunar Surface Classification — The Pareidolia Paradox", fontweight="bold", loc="left")

ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, color="#e5e7eb")
ax.set_axisbelow(True)

plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.1)
plt.savefig("results_chart_v2.png", dpi=220)
plt.show()