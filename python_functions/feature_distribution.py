from matplotlib import pyplot as plt
import seaborn as sns


def feature_distribution(dataset, features, cols=3):
    rows = (len(features) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.countplot(x=feature, hue=feature, data=dataset, ax=axes[i])
        axes[i].set_title(f'{feature} distribution')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')

    plt.tight_layout()
    plt.show()
