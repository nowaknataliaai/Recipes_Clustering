from Main import dane_skladniki, df_ograniczony_skladniki, przepisy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def wariancja_wyjasniona_pca(data, variables):
    pca = PCA()
    pca_dane = pca.fit(data).transform(data)
    procent_wariancji = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    skumulowana_wariancja = np.cumsum(procent_wariancji)
    labels = ['PC' + str(x) for x in range(1, len(procent_wariancji) + 1)]

    plt.figure(figsize=(16, 8))
    plt.bar(x=range(1, len(procent_wariancji) + 1), height=procent_wariancji, tick_label=labels, alpha=0.7,
            label='Wariancja %')
    plt.plot(range(1, len(procent_wariancji) + 1), procent_wariancji, marker='o', color='red')
    plt.plot(range(1, len(skumulowana_wariancja) + 1), skumulowana_wariancja, marker='x', linestyle="--", color='green', label='Skumulowana wariancja %')
    plt.xlabel('Poszczególne komponenty')
    plt.ylabel('Procent wariancji wyjaśnionej')
    plt.yticks(np.arange(0, 105, 10))
    plt.title('Wyjaśniona wariancja przez główne składowe')
    plt.legend()
    plt.grid(alpha=0.3)
    nazwa = input("Podaj nazwę pliku do zapisu wykresu: ")
    plt.savefig(f'PCAs/wariancja_{nazwa}.jpg')
    plt.show()

    pca_df = pd.DataFrame(pca_dane, index=variables, columns=labels)
    plt.figure(figsize=(16, 8))
    plt.scatter(pca_df.PC1, pca_df.PC2)
    for i, podpis in enumerate(variables):
        plt.text(pca_dane[i, 0], pca_dane[i, 1], podpis, fontsize=6)

    plt.title("Wyniki PCA przepisów")
    plt.xlabel(f"PC1 - {procent_wariancji[0]}%")
    plt.ylabel(f"PC2 - {procent_wariancji[1]}%")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.grid(alpha=0.3)
    nazwa = input("Podaj nazwę pliku do zapisu wykresu: ")
    plt.savefig(f'PCAs/PCA_{nazwa}.jpg')
    plt.show()

scaler = StandardScaler()

zeskalowane_dane = scaler.fit_transform(dane_skladniki)
wariancja_wyjasniona_pca(zeskalowane_dane, przepisy)

zeskalowany_ograniczony = scaler.fit_transform(df_ograniczony_skladniki)
wariancja_wyjasniona_pca(zeskalowany_ograniczony, przepisy)
