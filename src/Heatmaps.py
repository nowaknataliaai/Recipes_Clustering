from Main import dane, df_binarny, df_ograniczony
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


def heatmap_plot(data):
    plt.figure(figsize=(15, 9))
    if all(data.iloc[:, 2:].dtypes == bool):
        sns.heatmap(data.iloc[:, 2:], annot=False, cbar=False)
    else:
        sns.heatmap(data.iloc[:, 2:], annot=False, norm= LogNorm())
    plt.title("Heatmapa składników przepisów")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    nazwa = input("Podaj nazwę pliku do zapisu wykresu: ")
    plt.savefig(f'Heatmaps/{nazwa}.jpg')
    plt.show()

heatmap_plot(dane)
heatmap_plot(df_binarny)
heatmap_plot(df_ograniczony)
