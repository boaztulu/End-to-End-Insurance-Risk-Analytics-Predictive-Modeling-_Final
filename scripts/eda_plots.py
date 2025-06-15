import matplotlib.pyplot as plt

def plot_histogram(df, column, bins=50, figsize=(8,4), save_path=None):
    plt.figure(figsize=figsize)
    df[column].hist(bins=bins)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_bar_counts(df, column, top_n=10, figsize=(8,4), save_path=None):
    top = df[column].value_counts().nlargest(top_n)
    plt.figure(figsize=figsize)
    top.plot(kind='bar')
    plt.title(f'Top {top_n} {column} Counts')
    plt.xlabel(column)
    plt.ylabel('Count')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_scatter(df, x, y, figsize=(6,6), save_path=None):
    plt.figure(figsize=figsize)
    plt.scatter(df[x], df[y], alpha=0.3)
    plt.title(f'{y} vs {x}')
    plt.xlabel(x)
    plt.ylabel(y)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_box(df, column, figsize=(8,3), save_path=None):
    plt.figure(figsize=figsize)
    plt.boxplot(df[column].dropna(), vert=False)
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_loss_ratio(df, group_col, figsize=(10,4), save_path=None):
    lr = (
        df.groupby(group_col)
          .apply(lambda x: x.TotalClaims.sum() / x.TotalPremium.sum())
          .reset_index(name='loss_ratio')
          .sort_values('loss_ratio')
    )
    plt.figure(figsize=figsize)
    plt.bar(lr[group_col].astype(str), lr['loss_ratio'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(group_col)
    plt.ylabel('Loss Ratio')
    plt.title(f'Loss Ratio by {group_col}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_time_series(df, x, y, figsize=(8,4), save_path=None):
    plt.figure(figsize=figsize)
    plt.plot(df[x], df[y], marker='o')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{y} over {x}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
