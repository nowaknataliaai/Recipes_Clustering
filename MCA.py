from Main import df_binarny_skladniki
import pandas as pd
import prince
import matplotlib.pyplot as plt


mca = prince.MCA(n_components=2)
wyniki = mca.fit(df_binarny_skladniki).transform(df_binarny_skladniki)
wyniki.columns = ["Wymiar 1", "Wymiar 2"]

plt.figure(figsize=(6, 5))
plt.scatter(wyniki["Wymiar 1"], wyniki["Wymiar 2"], s=20, color= 'red')
plt.title("Wyniki MCA dla występujących składników przepisów (Binary)")
plt.xlabel("Wymiar 1")
plt.ylabel("Wymiar 2")
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.axvline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
plt.tight_layout()
plt.grid(alpha = 0.3)
plt.savefig('Obraz_MCA.jpg')
plt.show()




