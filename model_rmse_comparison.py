import matplotlib.pyplot as plt
import seaborn as sns

# Model names and their mean RMSE from your results
models = ["Linear Regression", "Decision Tree", "Random Forest"]
rmse_means = [69204, 69195, 49403]

# Set style
sns.set_theme(style="whitegrid")

# Create figure
plt.figure(figsize=(8, 5))
barplot = sns.barplot(x=models, y=rmse_means, palette="coolwarm")

# Add values above bars
for i, value in enumerate(rmse_means):
    plt.text(i, value + 500, f"{value:,.0f}", ha='center', fontsize=10, fontweight='bold')

# Labels and title
plt.ylabel("Mean RMSE", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.title("Model Performance (10-Fold Cross-Validation RMSE)", fontsize=14, fontweight='bold')

# Remove spines for cleaner look
sns.despine()

# Show plot
plt.tight_layout()
plt.show()


