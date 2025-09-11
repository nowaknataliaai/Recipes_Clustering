from Main import dane, df_binarny, df_ograniczony
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(15,9))
sns.heatmap(dane.iloc[:, 2:], annot=False)
plt.title("Heatmapa składników przepisów")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'Heatmaps/oryginal.jpg')
plt.show()

plt.figure(figsize=(15,9))
sns.heatmap(df_binarny.iloc[:, 2:], annot=False)
plt.title("Heatmapa binarna składników przepisów")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'Heatmaps/binary.jpg')
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df_ograniczony.iloc[:, 2:], annot=False)
plt.title("Heatmapa 5 wybranych składników przepisów")
plt.savefig(f'Heatmaps/special.jpg')
plt.show()
