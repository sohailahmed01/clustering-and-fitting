import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import linregress, skew, kurtosis

def plot_relational_plot(df):
    plt.figure(figsize=(12,7))
    sns.scatterplot(data=df,x='GDP per capita',y='Score',hue='Score',palette='plasma',edgecolor='black',s=120)
    plt.title('GDP per Capita vs Happiness Score (2019)',fontsize=16,fontweight='bold')
    plt.xlabel('GDP per Capita',fontsize=14)
    plt.ylabel('Happiness Score',fontsize=14)
    plt.grid(True,linestyle='--',alpha=0.6)
    plt.savefig('relational_plot.png',dpi=300,bbox_inches='tight')
    plt.show()

def plot_categorical_plot(df):
    top_10=df.nlargest(10,'Score')
    plt.figure(figsize=(12,7))
    sns.barplot(x='Score',y='Country or region',data=top_10,palette='mako')
    plt.title('Top 10 Happiest Countries (2019)',fontsize=16,fontweight='bold')
    plt.xlabel('Happiness Score',fontsize=14)
    plt.ylabel('Country',fontsize=14)
    plt.grid(axis='x',linestyle='--',alpha=0.6)
    plt.savefig('categorical_plot.png',dpi=300,bbox_inches='tight')
    plt.show()

def plot_statistical_plot(df):
    plt.figure(figsize=(12,8))
    numeric_cols=df.select_dtypes(include=np.number)
    sns.heatmap(numeric_cols.corr(),annot=True,cmap='coolwarm',fmt=".2f",linewidths=1,linecolor='black',square=True,cbar_kws={'shrink':0.75})
    plt.title('Feature Correlation Heatmap',fontsize=16,fontweight='bold')
    plt.savefig('statistical_plot.png',dpi=300,bbox_inches='tight')
    plt.show()

def perform_clustering(df,col1,col2):
    scaler=StandardScaler()
    data=scaler.fit_transform(df[[col1,col2]])
    inertias=[]
    silhouette_scores=[]
    K=range(2,8)
    for k in K:
        kmeans=KMeans(n_clusters=k,random_state=42,n_init=10)
        labels=kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data,labels))
    plt.figure(figsize=(12,7))
    plt.plot(K,inertias,marker='o',linestyle='-',color='royalblue',markersize=10,markerfacecolor='red',linewidth=3)
    plt.title('Elbow Method for Optimal Cluster Number',fontsize=16,fontweight='bold')
    plt.xlabel('Number of Clusters',fontsize=14)
    plt.ylabel('Inertia',fontsize=14)
    plt.grid(True,linestyle='--',alpha=0.6)
    plt.savefig('elbow_plot.png',dpi=300,bbox_inches='tight')
    plt.show()
    optimal_k=K[np.argmax(silhouette_scores)]
    print(f"Optimal clusters: {optimal_k} (Silhouette Score: {max(silhouette_scores):.2f})")
    kmeans=KMeans(n_clusters=optimal_k,random_state=42,n_init=10)
    labels=kmeans.fit_predict(data)
    centers=scaler.inverse_transform(kmeans.cluster_centers_)
    return labels,data,centers[:,0],centers[:,1]

def plot_clustered_data(labels,data,x_centers,y_centers):
    plt.figure(figsize=(12,7))
    plt.scatter(data[:,0],data[:,1],c=labels,cmap='viridis',alpha=0.8,edgecolors='black',s=100)
    plt.scatter(x_centers,y_centers,marker='X',s=350,c='red',edgecolor='black',linewidth=2.5,label='Centroids')
    plt.xlabel('GDP per Capita (scaled)',fontsize=14)
    plt.ylabel('Social Support (scaled)',fontsize=14)
    plt.title('Country Clusters by GDP & Social Support',fontsize=16,fontweight='bold')
    plt.legend()
    plt.grid(True,linestyle='--',alpha=0.6)
    plt.savefig('clustering.png',dpi=300,bbox_inches='tight')
    plt.show()

def perform_fitting(df,x_col,y_col):
    scaler=StandardScaler()
    X=scaler.fit_transform(df[[x_col]])
    y=df[y_col].values
    slope,intercept,r_value,p_value,std_err=linregress(X.flatten(),y)
    x_fit=np.linspace(X.min(),X.max(),100)
    y_fit=slope*x_fit+intercept
    x_fit_orig=scaler.inverse_transform(x_fit.reshape(-1,1))
    return (scaler.inverse_transform(X),y,x_fit_orig.flatten(),y_fit)

def plot_fitted_data(X,y,x_fit,y_fit):
    plt.figure(figsize=(12,7))
    plt.scatter(X,y,alpha=0.7,label='Actual Data',edgecolors='black',s=100)
    plt.plot(x_fit,y_fit,color='red',linewidth=3,label='Regression Line')
    plt.xlabel('GDP per Capita',fontsize=14)
    plt.ylabel('Happiness Score',fontsize=14)
    plt.title('Linear Regression: GDP vs Happiness',fontsize=16,fontweight='bold')
    plt.legend()
    plt.grid(True,linestyle='--',alpha=0.6)
    plt.savefig('fitting.png',dpi=300,bbox_inches='tight')
    plt.show()

def main():
    df=pd.read_csv('data.csv')
    print("Dataset Head:\n",df.head())
    print("\nDataset Tail:\n",df.tail())
    print("\nDescriptive Statistics:\n",df.describe())
    numeric_df=df.select_dtypes(include=np.number)
    print("\nStatistical Metrics:")
    print(f"Mean Values:\n{numeric_df.mean()}")
    print(f"\nStandard Deviation:\n{numeric_df.std()}")
    print(f"\nSkewness:\n{numeric_df.apply(skew)}")
    print(f"\nKurtosis:\n{numeric_df.apply(kurtosis)}")
    print("\nCorrelation Matrix:\n",numeric_df.corr())
    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)
    clustering_results=perform_clustering(df,'GDP per capita','Social support')
    plot_clustered_data(*clustering_results)
    fitting_results=perform_fitting(df,'GDP per capita','Score')
    plot_fitted_data(*fitting_results)

if __name__=='__main__':
    main()
