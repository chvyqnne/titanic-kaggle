import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(df):
    print("DataFrame Head:")
    print(df.head())
      
    print("\nDataFrame Shape:")
    print(df.shape)
      
    print("\nDataFrame Description:")
    print(df.describe())
      
    print("\nNull Values in DataFrame:")
    print(df.isnull().sum())
      
    print("\nUnique Values in Each Column:")
    print(df.nunique())

def preprocess_data(df):
    df = df.copy()

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    df['Age_Missing'] = df['Age'].isnull().astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
    
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
    
    return df

def plot_bar(ax, df, column):
    df[column].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'{column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

def visualize_bar_charts(df):
    columns = ['Survived', 'Pclass', 'Sex', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch']
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    for i, column in enumerate(columns):
        plot_bar(axes[i], df, column)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_histogram(df, column):
    plt.figure(figsize=(10, 6))
    df[column].hist(bins=30)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
                
def visualize_histograms(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        axes[i].hist(df[col].dropna(), bins=30)
        axes[i].set_title(f'Histogram of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j]) 

    plt.tight_layout()
    plt.show()

def pairplot(df):
    sns.pairplot(df, hue='Survived', vars=['Age', 'Fare', 'Sex', 'Pclass'])
    plt.show()

def barplot(df, x, y):
    sns.barplot(x=x, y=y, data=df)
    plt.show()

def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def main():
    df = pd.read_csv('data/train.csv')
    
    print("\n=== Raw Data Analysis ===")
    analyze_data(df)

    print("\n=== Preprocessing Data ===")
    train_data = preprocess_data(df)

    print("\n=== Visualizing Categorical Distributions ===")
    visualize_bar_charts(train_data)

    print("\n=== Visualizing Histograms ===")
    visualize_histograms(train_data)

    print("\n=== Pair Plot ===")
    pairplot(train_data)

    print("\n=== Survival Rate by Pclass ===")
    barplot(train_data, x='Pclass', y='Survived')

    print("\n=== Correlation Matrix ===")
    plot_correlation_matrix(train_data)

if __name__ == '__main__':
    main()