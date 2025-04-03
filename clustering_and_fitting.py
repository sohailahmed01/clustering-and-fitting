import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def plot_relational_plot(df):
    """Generates a scatter plot of GDP per capita against Happiness Score"""
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df, x='GDP per capita', y='Score', hue='Score',
        palette='plasma', edgecolor='black', s=120
    )
    plt.title('GDP per Capita vs Happiness Score (2019)',
              fontsize=16, fontweight='bold')
    plt.xlabel('GDP per capita', fontsize=14)
    plt.ylabel('Happiness Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('relational_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


def plot_categorical_plot(df):
    """Creates a bar plot of the top 10 happiest countries"""
    top_10 = df.nlargest(10, 'Score')
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Score', y='Country or region', data=top_10, 
                palette='viridis')
    plt.title('Top 10 Happiest Countries (2019)', fontsize=16,
              fontweight='bold')
    plt.xlabel('Happiness Score', fontsize=14)
    plt.ylabel('Country', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.savefig('categorical_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


def plot_statistical_plot(df):
    """Produces a heatmap showing the correlation matrix of numeric features"""
    numeric_cols = df.select_dtypes(include=np.number)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f",
        linewidths=1, linecolor='black', square=True, cbar_kws={'shrink': 0.75}
    )
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.savefig('statistical_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


def statistical_analysis(df, col: str):
    """Computes statistical moments for a specified column"""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Performs data preprocessing and displays dataset summaries"""
    print("Dataset Head:\n", df.head())
    print("\nDataset Tail:\n", df.tail())
    print("\nDescriptive Statistics:\n", df.describe())
    numeric_df = df.select_dtypes(include=np.number)
    print("\nStatistical Metrics:")
    print(f"Mean Values:\n{numeric_df.mean()}")
    print(f"\nStandard Deviation:\n{numeric_df.std()}")
    print(f"\nSkewness:\n{numeric_df.apply(ss.skew)}")
    print(f"\nKurtosis:\n{numeric_df.apply(ss.kurtosis)}")
    print("\nCorrelation Matrix:\n", numeric_df.corr())
    return df


def writing(moments, col):
    """Prints formatted statistical analysis results"""
    mean, std, skew, kurt = moments
    print(f'For the attribute {col}:')
    print(f'Mean = {mean:.2f}, Standard Deviation = {std:.2f},'
          f'Skewness = {skew:.2f}, and Excess Kurtosis = {kurt:.2f}.')
    skew_dir = "right" if skew > 0 else "left" if skew < 0 else "not"
    if kurt > 0:
      kurt_type = "leptokurtic"
    elif kurt < 0:
      kurt_type = "platykurtic"
    else:
      kurt_type = "mesokurtic"
    print(f'The data was {skew_dir} skewed and {kurt_type}.')
    return

def perform_clustering(df, col1, col2):
    """Executes K-means clustering and determines optimal cluster count"""
    scaler = StandardScaler()
    data = scaler.fit_transform(df[[col1, col2]])
    inertias = []
    silhouette_scores = []
    K = range(2, 8)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))
    
    def plot_elbow_method():
        plt.figure(figsize=(12, 7))
        plt.plot(
            K, inertias,marker='o', linestyle='-', color='royalblue',
            markersize=10,markerfacecolor='red',linewidth=3
        )
        plt.title('Elbow Method for Optimal Cluster Number',fontsize=16,fontweight='bold')
        plt.xlabel('Number of Clusters', fontsize=14)
        plt.ylabel('Inertia', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
    def one_silhouette_inertia():
        optimal_k = K[np.argmax(silhouette_scores)]
        _score = max(silhouette_scores)
        _inertia = inertias[optimal_k - 2]
        return _score, _inertia
    score, inertia = one_silhouette_inertia()
    plot_elbow_method()
    
    optimal_k = K[np.argmax(silhouette_scores)]
    print(f"Optimal clusters: {optimal_k} (Silhouette Score: {score:.2f})")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans = centers[:, 0]
    ykmeans = centers[:, 1]
    cenlabels = list(range(optimal_k))
    
    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Plots clustered data points and centroids"""
    plt.figure(figsize=(12, 7))
    plt.scatter(
        data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.8,
        edgecolors='black', s=100, label='Data Points'
    )
    plt.scatter(
        xkmeans, ykmeans, marker='X', s=350, c='red', edgecolor='black',
        linewidth=2.5, label='Centroids'
    )
    for i, label in enumerate(centre_labels):
        plt.text(
            xkmeans[i], ykmeans[i], str(label),
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )
    plt.xlabel('GDP per Capita (scaled)', fontsize=14)
    plt.ylabel('Social Support (scaled)', fontsize=14)
    plt.title('Country Clusters by GDP & Social Support',fontsize=16,
              fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


def perform_fitting(df, col1, col2):
    """Fits a linear regression model between two features"""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[[col1]])
    y = df[col2].values
    slope, intercept, r_value, p_value,std_err = ss.linregress(X.flatten(), y)
    x_fit = np.linspace(X.min(), X.max(), 100)
    y_fit = slope * x_fit + intercept
    x_fit_orig = scaler.inverse_transform(x_fit.reshape(-1, 1)).flatten()
    X_orig = scaler.inverse_transform(X)
    data = np.column_stack((X_orig.flatten(), y))
    return data, x_fit_orig, y_fit


def plot_fitted_data(data, x, y):
    """Visualizes the linear regression fit"""
    plt.figure(figsize=(12, 7))
    plt.scatter(
        data[:, 0], data[:, 1], alpha=0.7,
        edgecolors='black', s=100, label='Actual Data'
    )
    plt.plot(x, y, color='red', linewidth=3, label='Regression Line')
    plt.xlabel('GDP per Capita', fontsize=14)
    plt.ylabel('Happiness Score', fontsize=14)
    plt.title('Linear Regression: GDP vs Happiness',
              fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('fitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'GDP per capita'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    clustering_results=perform_clustering(df,'GDP per capita', 'Social support')
    plot_clustered_data(*clustering_results)
    fitting_results = perform_fitting(df, 'GDP per capita', 'Score')
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()
