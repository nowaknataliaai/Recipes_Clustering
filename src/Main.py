import pandas as pd
import seaborn as sns
import numpy as np

sciezka = "C:/Users/natalia.nowak/Desktop/Klasteryzacja/Klasteryzacja_przepisow/.venv/przepisy.xlsx"
#sciezka = "C:/Users/natalia.nowak/Desktop/Projekt BC/Breast_Cancer_Data_Analysis/Breast_Cancer_Wisconsin_Diagnostic_dataset.xlsx"
dane = pd.read_excel(sciezka)
kolumna_kategoryczna = "Nazwa"
ID = "Id"
przepisy = dane[kolumna_kategoryczna]
skladniki = pd.DataFrame(dane.columns.difference([ID, kolumna_kategoryczna]))
dane[dane.isna()] = 0

dane_skladniki = dane.drop(columns =["Id", kolumna_kategoryczna])

df_binarny = dane.copy()
kategoryczne = df_binarny.iloc[:, :2]
cechy = df_binarny.iloc[:, 2:].astype(bool)
df_binarny = pd.concat([kategoryczne, cechy], axis=1) #zeby nie bylo ostrzezen
df_binarny_skladniki = df_binarny.drop(columns=[ID, kolumna_kategoryczna])
#df_binarny.to_excel("przepisy_binarny.xlsx", index=False)

df_ograniczony = dane.iloc[:, [0, 1, 2, 3, 6, 8, 12]]
df_ograniczony_skladniki = df_ograniczony.drop(columns= [ID, kolumna_kategoryczna])
#df_ograniczony.to_excel("przepisy_ograniczony.xlsx", index=False)
