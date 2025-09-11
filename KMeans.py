from Main import dane, df_binarny_skladniki, df_ograniczony_skladniki, dane_skladniki
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

scaler = StandardScaler()
skalowane_dane = scaler.fit_transform(dane_skladniki)

inertias = []
max_k = 100
for k in range(1, max_k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(skalowane_dane)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
plt.plot( inertias, 'o-')
plt.xlabel('Liczba klastrów')
plt.ylabel('BezwŁadność (suma kwadratów odległości między centroidami)')
plt.title('Metoda łokcia')
plt.grid(True)
plt.savefig('KMeans/lokiec.jpg')
plt.show()

k = 5
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(skalowane_dane)
centroidy = kmeans.cluster_centers_
inertia = kmeans.inertia_

pca = PCA(n_components=2)
pca.fit(skalowane_dane)
x_pca = pca.transform(skalowane_dane)
centroidy_pca = pca.transform(centroidy)
procent_wariancji = np.round(pca.explained_variance_ratio_*100, 1)

plt.figure(figsize=(8,6))
cmap = plt.get_cmap('tab20')
unique_labels = np.unique(labels)
for lab in unique_labels:
    plt.scatter( x_pca[labels == lab, 0], x_pca[labels == lab, 1], label = f'Klaster {lab+1}', c=np.array([cmap(lab % 10)]))

plt.scatter(centroidy_pca[:, 0], centroidy_pca[:, 1], s= 100, marker='X', edgecolor='k', label= 'Centroidy', c= [cmap(i % 10) for i in range(len(centroidy_pca))])

plt.xlabel(f'PC1 - {procent_wariancji[0]}%')
plt.ylabel(f'PC2 - {procent_wariancji[1]}%')
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.axvline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.title(f'KMeans (k = {k}) + PCA (2 składowe) \nBezwładność = {inertia:.2f}')
plt.legend(loc='best', fontsize= 'small')
plt.grid(alpha=0.3)
plt.savefig('KMeans/klasteryzacja.jpg')
plt.show()

#df binarny
skalowane_dane = scaler.fit_transform(df_binarny_skladniki)

inertias = []
max_k = 100
for k in range(1, max_k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(skalowane_dane)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
plt.plot( inertias, 'o-')
plt.xlabel('Liczba klastrów')
plt.ylabel('BezwŁadność (suma kwadratów odległości między centroidami)')
plt.title('Metoda łokcia')
plt.grid(True)
plt.savefig('KMeans/lokiec.jpg')
plt.show()

k = 8
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(skalowane_dane)
centroidy = kmeans.cluster_centers_
inertia = kmeans.inertia_

pca = PCA(n_components=2)
pca.fit(skalowane_dane)
x_pca = pca.transform(skalowane_dane)
centroidy_pca = pca.transform(centroidy)
procent_wariancji = np.round(pca.explained_variance_ratio_*100, 1)

plt.figure(figsize=(8,6))
cmap = plt.get_cmap('tab20')
unique_labels = np.unique(labels)
for lab in unique_labels:
    plt.scatter( x_pca[labels == lab, 0], x_pca[labels == lab, 1], label = f'Klaster {lab+1}', c=np.array([cmap(lab % 10)]))

plt.scatter(centroidy_pca[:, 0], centroidy_pca[:, 1], s= 100, marker='X', edgecolor='k', label= 'Centroidy', c= [cmap(i % 10) for i in range(len(centroidy_pca))])

plt.xlabel(f'PC1 - {procent_wariancji[0]}%')
plt.ylabel(f'PC2 - {procent_wariancji[1]}%')
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.axvline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.title(f'KMeans (k = {k}) + PCA (2 składowe) \nBezwładność = {inertia:.2f}')
plt.legend(loc='best', fontsize= 'small')
plt.grid(alpha=0.3)
plt.savefig('KMeans/klasteryzacja.jpg')
plt.show()

#df 5 skladnikow
skalowane_dane = scaler.fit_transform(df_ograniczony_skladniki)

inertias = []
max_k = 100
for k in range(1, max_k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(skalowane_dane)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
plt.plot( inertias, 'o-')
plt.xlabel('Liczba klastrów')
plt.ylabel('BezwŁadność (suma kwadratów odległości między centroidami)')
plt.title('Metoda łokcia')
plt.grid(True)
plt.savefig('KMeans/lokiec.jpg')
plt.show()

k = 9
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(skalowane_dane)
centroidy = kmeans.cluster_centers_
inertia = kmeans.inertia_

pca = PCA(n_components=2)
pca.fit(skalowane_dane)
x_pca = pca.transform(skalowane_dane)
centroidy_pca = pca.transform(centroidy)
procent_wariancji = np.round(pca.explained_variance_ratio_*100, 1)

plt.figure(figsize=(8,6))
cmap = plt.get_cmap('tab20')
unique_labels = np.unique(labels)
for lab in unique_labels:
    plt.scatter( x_pca[labels == lab, 0], x_pca[labels == lab, 1], label = f'Klaster {lab+1}', c=np.array([cmap(lab % 10)]))

plt.scatter(centroidy_pca[:, 0], centroidy_pca[:, 1], s= 100, marker='X', edgecolor='k', label= 'Centroidy', c= [cmap(i % 10) for i in range(len(centroidy_pca))])

plt.xlabel(f'PC1 - {procent_wariancji[0]}%')
plt.ylabel(f'PC2 - {procent_wariancji[1]}%')
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.axvline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.title(f'KMeans (k = {k}) + PCA (2 składowe) \nBezwładność = {inertia:.2f}')
plt.legend(loc='best', fontsize= 'small')
plt.grid(alpha=0.3)
plt.savefig('KMeans/klasteryzacja.jpg')
plt.show()