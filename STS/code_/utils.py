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
    fig.tight_layout()
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


def print_evaluation(specific, overall, validation = False, is_train = True):
    specific.set_index('Dataset', inplace = True)
    overall.set_index('Dataset', inplace = True)
    print('-'*80)

    if is_train:
        print('Evaluation on TRAINING')
    else:
        if validation:
            print('Evaluation on VALIDATION')
        else:
            print('Evaluation on TEST')
    print()

    print('Pearson Correlation for each specific dataset:')
    print(specific)
    print()
    if is_train:
        print('Average Pearson Correlation over all TRAIN CV:')
    else:
        if validation:
            print('Overall Pearson Correlation for the concatenation of the VALIDATION datasets:')
        else:
            print('Overall Pearson Correlation for the concatenation of the TEST datasets:')

    print(overall)
    print('-'*80)


def plot_importances(importances, std, feature_names, name, print_error = True):
    importances = pd.DataFrame(importances, index=feature_names)
    importances['Average'] = importances.mean(axis=1)

    # Put average on the first column
    importances.insert(0, 'Average', importances.pop('Average'))

    # sort dataframe
    importances.sort_values(by=['Average'], inplace = True)

    # sort standard deviations as well with the same index
    std_df = pd.DataFrame(std, index=feature_names)
    std_df.reindex(importances.index)
    std_df['Average'] = std_df.mean(axis=1)
    std_df.insert(0, 'Average', std_df.pop('Average'))

    # cut the names out since they are quite long
    #if name != 'SVR':
        #feature_names = [name[:min(10, len(name))] for name in feature_names]
        #importances.reindex(feature_names)
        #std_df.reindex(feature_names)
    if name == 'SVR':
        std_df.drop(columns=['Average'], inplace=True)
        importances.drop(columns=['Average'], inplace=True)

    if name == 'SVR':
        fig, ax = plt.subplots()
        importances.plot.bar(yerr = std_df, ax = ax) 
        ax.tick_params(axis='x', which='major', labelsize=10, labelrotation=0)
        ax.get_legend().remove()
    else:
        fig, ax = plt.subplots(figsize = (8, 20))
        if print_error: 
            importances.plot.barh(xerr = std_df, ax = ax, error_kw=dict(elinewidth=0.1), width = 0.95) 
        else:
            importances.plot.barh(ax = ax, width = 0.95) 
        ax.tick_params(axis='y', which='major', labelsize=5)

    if name == 'impurity':
        ax.set_title("Feature importances using MDI")
        ax.set_xlabel("Mean decrease in impurity")
    else:
        ax.set_title("Feature importances using permutation")
        ax.set_xlabel("Mean Pearson decrease")
    fig.tight_layout()

    plt.savefig(f'{name}.png', dpi=1000, bbox_inches='tight', pad_inches=0)


def plot_feature_selection(scores, num_scores = 50):
    fig, ax = plt.subplots(figsize = (14, 8))
    for model, (avg, std) in scores.items():
        ax.errorbar(x = range(1, num_scores + 1), y = avg[:num_scores], yerr = std[:num_scores], label = model.title())
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel("Mean test Pearson")
    ax.set_title("Recursive Feature Elimination \nwith correlated features")
    ax.legend()
    plt.savefig('selection.png', dpi=1000, bbox_inches='tight', pad_inches=0)
