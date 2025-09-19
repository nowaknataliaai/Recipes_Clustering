from Main import df_binarny_skladniki, przepisy
import pandas as pd
import prince
import matplotlib.pyplot as plt


def Mca_plot(data, variables):
    mca = prince.MCA(n_components=2)
    results = mca.fit(data).transform(data)
    results.columns = ["PC1", "PC2"]
    procent_wariancji_1 = round(mca.percentage_of_variance_[0], 2)
    procent_wariancji_2 = round(mca.percentage_of_variance_[1], 2)

    plt.figure(figsize=(16, 8))
    plt.scatter(results["PC1"], results["PC2"], s=20, color='red')
    for i, podpis in enumerate(variables):
        plt.text(results.iloc[i, 0], results.iloc[i, 1], podpis, fontsize=6)
    plt.title("Wyniki MCA dla występujących składników przepisów (Binary)")
    plt.xlabel(f"PC1 - {procent_wariancji_1}%")
    plt.ylabel(f"PC2 - {procent_wariancji_2}%")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig('Obraz_MCA.jpg')
    plt.show()

Mca_plot(df_binarny_skladniki, przepisy)
