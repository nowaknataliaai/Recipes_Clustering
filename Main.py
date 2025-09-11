import pandas as pd
import seaborn as sns
import numpy as np


dane = pd.read_excel("C:/Users/natalia.nowak/Desktop/Klasteryzacja/Klasteryzacja_przepisow/.venv/przepisy.xlsx")
kolumna_kategoryczna = "Nazwa"
przepisy = dane[kolumna_kategoryczna]
skladniki = pd.DataFrame(dane.columns.difference(["Id", kolumna_kategoryczna]))
dane[dane.isna()] = 0

df_binarny = dane.copy()
dane_skladniki = dane.drop(columns =["Id", kolumna_kategoryczna])
df_binarny[dane_skladniki == 0] = 0
df_binarny[dane_skladniki != 0] = 1
df_binarny_skladniki = df_binarny.drop(columns=["Id", kolumna_kategoryczna])
#df_binarny.to_excel("przepisy_binarny.xlsx", index=False)

df_ograniczony = dane.iloc[:, [0, 1, 2, 3, 6, 8, 12]]
df_ograniczony_skladniki = df_ograniczony.drop(columns= ["Id", kolumna_kategoryczna])
#df_ograniczony.to_excel("przepisy_ograniczony.xlsx", index=False)
