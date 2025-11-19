from matplotlib import pyplot as plt
import seaborn as sns


def describe(data):
    return data.describe()


def corr_matrix(data):
    corr = data.corr()
    plt.figure(figsize=(22, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()
