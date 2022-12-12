import seaborn as sns
import pandas as pd
import numpy as np
import pca as pca
import matplotlib.pyplot as plt


def covariance_matrix_plot(x, y):
    cmap = sns.diverging_palette(500, 10, as_cmap=True)
    df = x.copy()
    df['Gold Standard'] = y.copy()
    df = df.corr()
    sns.set(font_scale=0.2)
    sns.set_style(style = 'white')
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    plot = sns.heatmap(df, mask=mask, cmap=cmap, square=True, linewidth=.5, cbar_kws = {"shrink": .5}, cbar = False)
    fig = plot.get_figure()
    fig.savefig('correlation.png', dpi=1000, bbox_inches='tight', pad_inches=0)

def PCA(x, y):
    sns.set(font_scale=1)
    sns.set_style(style = 'white')
    model = pca.pca(n_components=10)
    results = model.fit_transform(x) # Fit transform
    #fig, ax = model.plot() # Plot explained variance
    #fig.savefig('explained.png', dpi=1000, bbox_inches='tight', pad_inches=0)
    cmap = sns.color_palette("magma", as_cmap=True)
    fig, ax = model.biplot(y = y, n_feat = 20, label = None, fontdict = {'color': 'red', 'size': 15}, cmap = cmap, color_arrow = 'blue', legend = False, visible = True) # Make biplot with the 10 that explain the most
    norm = plt.Normalize(y.min(), y.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ax.figure.colorbar(sm)
    fig.savefig('biplot.png', dpi=1000, bbox_inches='tight', pad_inches=0)


def print_evaluation(specific, overall, is_train = True):
    specific.set_index('Dataset', inplace = True)
    overall.set_index('Dataset', inplace = True)

    print('-'*80)
    print('Pearson Correlation for each specific dataset:')
    print(specific)
    print()
    if is_train:
        print('Average Pearson Correlation over all CV:')
    else:
        print('Overall Pearson Correlation for the concatenation of the datasets:')
    print(overall)
    print('-'*80)
