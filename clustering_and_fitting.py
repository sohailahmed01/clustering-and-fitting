"""
World Happiness Report Analysis (2019)
Clustering, Fitting, and Statistical Analysis
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import linregress

def plot_relational_plot(df):
    """Create a scatter plot of GDP vs Happiness Score."""
    fig,ax=plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df,x='GDP per capita',y='Score',hue='Score',palette='viridis')
    plt.title('GDP per Capita vs Happiness Score (2019)')
    plt.savefig('relational_plot.png')
    plt.show()
    return

def plot_categorical_plot(df):
    """Create a horizontal bar plot of top 10 countries."""
    top_10=df.nlargest(10,'Score')
    plt.figure(figsize=(10,6))
    sns.barplot(x='Score',y='Country or region',data=top_10,palette='Blues_d')
    plt.title('Top 10 Happiest Countries (2019)')
    plt.savefig('categorical_plot.png')
    plt.show()
    return

def plot_statistical_plot(df):
    """Create a correlation heatmap."""
    plt.figure(figsize=(12,8))
    numeric_cols=df.select_dtypes(include=np.number)
    sns.heatmap(numeric_cols.corr(),annot=True,cmap='coolwarm',fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.savefig('statistical_plot.png')
    plt.show()
    return

def statistical_analysis(df, col:str):
    """Calculate statistical moments."""
    mean=df[col].mean()
    stddev=df[col].std()
    skew=df[col].skew()
    excess_kurtosis=df[col].kurtosis()
    return mean,stddev,skew,excess_kurtosis

def preprocessing(df):
    """Clean and prepare data."""
    df=df.dropna().reset_index(drop=True)
    print("\nData preview:")
    print(df[['Country or region','Score','GDP per capita']].head(3))
    return df

def writing(moments, col):
    """Interpret statistical results."""
    print(f'\nStatistical analysis for {col}:')
    print(f'Mean = {moments[0]:.2f}, Std Dev = {moments[1]:.2f}')
    print(f'Skewness = {moments[2]:.2f}, Excess Kurtosis = {moments[3]:.2f}')
    
    # Interpret results
    skew="Right" if moments[2] > 0.5 else "Left" if moments[2] < -0.5 else "Mild"
    kurt="Leptokurtic" if moments[3] > 1 else "Platykurtic" if moments[3] < -1 else "Mesokurtic"
    print(f'Distribution: {skew}-skewed,{kurt}')

def perform_clustering(df,col1,col2):
    """Perform K-Means clustering with numerical columns."""
    # Validate numerical columns
    assert col1 in df.select_dtypes(include=np.number).columns
    assert col2 in df.select_dtypes(include=np.number).columns
    
    scaler=StandardScaler()
    data=scaler.fit_transform(df[[col1,col2]])
    
    # Elbow method
    inertias=[]
    sil_scores=[]
    K=range(2, 8)
    
    for k in K:
        kmeans=KMeans(n_clusters=k,random_state=42)
        labels=kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(data, labels))
    
    # Plot elbow
    plt.figure(figsize=(10,6))
    plt.plot(K,inertias,'bo-')
    plt.title('Elbow Method for Optimal Cluster Number')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.savefig('elbow_plot.png')
    plt.show()
    
    # Choose optimal k (3 based on elbow plot)
    optimal_k=3
    kmeans=KMeans(n_clusters=optimal_k,random_state=42)
    labels=kmeans.fit_predict(data)
    centers=scaler.inverse_transform(kmeans.cluster_centers_)
    
    return labels, data,centers[:, 0],centers[:, 1],kmeans.cluster_centers_

def plot_clustered_data(labels,data, x_centers,y_centers,centers):
    """Visualize clustering results."""
    plt.figure(figsize=(10,6))
    plt.scatter(data[:,0],data[:,1],c=labels,cmap='viridis',alpha=0.6)
    plt.scatter(x_centers,y_centers,marker='X',s=200,c='red',label='Centroids')
    plt.xlabel('GDP per capita (scaled)')
    plt.ylabel('Social Support (scaled)')
    plt.title('Country Clusters by GDP & Social Support')
    plt.legend()
    plt.savefig('clustering.png')
    plt.show()

def perform_fitting(df,x_col,y_col):
    """Perform linear regression."""
    x=df[x_col].values
    y=df[y_col].values
    slope,intercept,r_value,p_value,std_err=linregress(x,y)
    y_pred=slope * x + intercept
    
    print(f'\nRegression Results ({x_col} vs {y_col}):')
    print(f'Slope: {slope:.2f},Intercept: {intercept:.2f}')
    print(f'R²: {r_value**2:.2f},p-value: {p_value:.4e}')
    
    return (x,y),x,y_pred

def plot_fitted_data(data,x,y_pred):
    """Plot regression line."""
    plt.figure(figsize=(10,6))
    plt.scatter(data[0],data[1],alpha=0.6,label='Actual Data')
    plt.plot(x,y_pred,color='red',linewidth=2,label='Regression Line')
    plt.xlabel('GDP per capita')
    plt.ylabel('Happiness Score')
    plt.title('Linear Regression: GDP vs Happiness')
    plt.legend()
    plt.savefig('fitting.png')
    plt.show()

def main():
    df=pd.read_csv('data.csv')
    df=preprocessing(df)
    
    # Visualizations
    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)
    
    # Statistical analysis
    moments = statistical_analysis(df,'Score')
    writing(moments,'Happiness Score')
    
    # Clustering analysis
    clustering_results=perform_clustering(df,'GDP per capita','Social support')
    plot_clustered_data(*clustering_results)
    
    # Regression analysis
    fitting_results=perform_fitting(df,'GDP per capita','Score')
    plot_fitted_data(*fitting_results)

if __name__ == '__main__':
    main()
