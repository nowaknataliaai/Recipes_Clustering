from Main import dane_skladniki, df_ograniczony_skladniki
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#ORYGINALNY DF Z PRZEPISAMI
scaler = StandardScaler()
zeskalowane_dane = scaler.fit_transform(dane_skladniki)
pca = PCA()
pca.fit(zeskalowane_dane)
pca_dane = pca.transform(zeskalowane_dane)
procent_wariancji = np.round(pca.explained_variance_ratio_*100,  decimals=1)
skumulowana_wariancja = np.cumsum(procent_wariancji)
labels = ['PC' + str(x) for x in range(1, len(procent_wariancji)+1)]

plt.figure(figsize=(16, 8))
plt.bar(x=range(1, len(procent_wariancji)+1), height=procent_wariancji, tick_label = labels, alpha = 0.7, label = 'Wariancja %')
plt.plot(range(1, len(procent_wariancji)+1), procent_wariancji, marker = 'o', color = 'red')
plt.plot(range(1, len(skumulowana_wariancja)+1), skumulowana_wariancja, marker = 'x', linestyle = "--", color = 'green', label = 'Skumulowana wariancja %')
plt.xlabel('Poszczególne komponenty')
plt.ylabel('Procent wariancji wyjaśnionej')
plt.yticks(np.arange(0, 105, 10))
plt.title('Wyjaśniona wariancja przez główne składowe')
plt.legend()
plt.grid(alpha = 0.3)
plt.savefig(f'PCAs/wariancja_original.jpg')
plt.show()

pca_df = pd.DataFrame(pca_dane, index = dane_skladniki.index, columns = labels)

plt.figure(figsize=(16,8))
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title("Wyniki PCA przepisów")
plt.xlabel(f"PC1 - {procent_wariancji[0]}%")
plt.ylabel(f"PC2 - {procent_wariancji[1]}%")
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.axvline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.grid(alpha = 0.3)
plt.savefig(f'PCAs/PCA_original.jpg')
plt.show()

######################################################################################################################
#DF Z PRZEPISAMI DLA 5 WYBRANYCH SKŁADNIKÓW
zeskalowane_dane = scaler.fit_transform(df_ograniczony_skladniki)
pca = PCA()
pca.fit(zeskalowane_dane)
pca_dane = pca.transform(zeskalowane_dane)
procent_wariancji = np.round(pca.explained_variance_ratio_*100,  decimals=1)
skumulowana_wariancja = np.cumsum(procent_wariancji)
labels = ['PC' + str(x) for x in range(1, len(procent_wariancji)+1)]

plt.bar(x=range(1, len(procent_wariancji)+1), height=procent_wariancji, tick_label = labels, alpha = 0.7, label = 'Wariancja %')
plt.plot(range(1, len(procent_wariancji)+1), procent_wariancji, marker = 'o', color = 'red')
plt.plot(range(1, len(skumulowana_wariancja)+1), skumulowana_wariancja, marker = 'x', linestyle = "--", color = 'green', label = 'Skumulowana wariancja %')
plt.xlabel('Poszczególne komponenty')
plt.ylabel('Procent wariancji wyjaśnionej')
plt.yticks(np.arange(0, 105, 10))
plt.title('Wyjaśniona wariancja przez główne składowe')
plt.legend()
plt.grid(alpha = 0.3)
plt.savefig(f'PCAs/wariancja_special.jpg')
plt.show()

pca_df = pd.DataFrame(pca_dane, index = df_ograniczony_skladniki.index, columns = labels)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title("PCA przepisów z wybranymi 5 składnikami")
plt.xlabel(f"PC1 - {procent_wariancji[0]}%")
plt.ylabel(f"PC2 - {procent_wariancji[1]}%")
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.axvline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.grid(alpha = 0.3)
plt.savefig(f'PCAs/PCA_special.jpg')
plt.show()

