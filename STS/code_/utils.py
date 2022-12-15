from treeinterpreter import treeinterpreter as ti
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pca as pca


def covariance_matrix_plot(x, y):
    cmap = plt.cm.coolwarm_r
    df = x.copy()
    df['Gold Standard'] = y.copy()
    df = df.corr()
    plt.figure(figsize=(10, 10), dpi=1000)
    sns.set(font_scale=0.5)
    sns.set_style(style = 'white')
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    plot = sns.heatmap(df, mask=mask, cmap=cmap, square=True, linewidth=.5, cbar_kws = {"shrink": .5})
    fig = plot.get_figure()
    fig.tight_layout()
    plt.plot()
    #fig.savefig('correlation.png', dpi=1000, bbox_inches='tight', pad_inches=0)

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
    ax.figure.colorbar(sm) # shows correctly in savefig but not in plt.show()
    fig.savefig('PCA.png', dpi=1000, bbox_inches='tight', pad_inches=0)



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
        ax.tick_params(axis='y', which='major', labelsize=8)

    if name == 'impurity':
        ax.set_title("Feature importances using MDI")
        ax.set_xlabel("Mean decrease in impurity")
    else:
        ax.set_title("Feature importances using permutation")
        ax.set_xlabel("Mean Pearson decrease")
    fig.tight_layout()

    plt.show()
    #plt.savefig(f'{name}.png', dpi=1000, bbox_inches='tight', pad_inches=0)


def plot_feature_selection(scores):
    fig, ax = plt.subplots(figsize = (14, 8))
    num_scores = 1000
    for model, (avg, std) in scores.items():
        num_scores = min(num_scores, len(avg))

    for model, (avg, std) in scores.items():
        ax.errorbar(x = range(1, num_scores + 1), y = avg[:num_scores], yerr = std[:num_scores], label = model.title())
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel("Mean test Pearson")
    ax.set_title("Recursive Feature Elimination \nwith correlated features")
    ax.legend()
    plt.show()

    #plt.savefig('selection.png', dpi=1000, bbox_inches='tight', pad_inches=0)


def print_sentence(sentence, idx):
    print('Pair number:', idx, '- origin:', sentence['Origin'])
    print(f"\tDifference = {round(sentence['Gold Standard'] - sentence['Predicted'], 3)}, Gold Standard = {sentence['Gold Standard']}, Predicted = {round(sentence['Predicted'], 3)}")
    print('\tSentence 1:', sentence['Sentence 1'])
    print('\tSentence 2:', sentence['Sentence 2'])
    print()

def get_interpretations(data, sentences, true, regressors, x_features, feature_names, feature_subset):
    data_stored = {}
    for idx in data.index:
        sentence = sentences.loc[idx]
        model = sentence['Origin']
        
        x = x_features[idx == true.index].reshape(1, -1)      
        data_stored[idx] = np.zeros(len(feature_names))
        if len(feature_subset) == 0:
            _, _, contributions = ti.predict(regressors[model].best_estimator_, x)
            data_stored[idx] = contributions[0]
        else:
            _, _, contributions = ti.predict(regressors[model].best_estimator_, x[:, feature_subset[model]])
            data_stored[idx][feature_subset[model]] = contributions[0]

    return data_stored
            

def interpretations_plot(data, name, feature_names):
    data = pd.DataFrame(data, index=feature_names)

    # remove all zero rows
    data = data.loc[~(data == 0).all(axis=1)] 

    # Filter some that are not so important
    if name == 'best':
        data = data.loc[~(data.abs() < 0.005).all(axis=1)] 
    else:
        data = data.loc[~(data.abs() < 0.005).all(axis=1)] 

    ax = data.plot.barh(width = 0.95, subplots=True, figsize = (8, 12)) 
    ax[0].set_title(f"Feature Contribution for {name} Predictions")
    ax[1].set_title("")
    ax[2].set_title("")
    ax[2].set_xlabel("Feature Contribution to Prediction")

    plt.show()

def sentence_analysis(predicted, true, sentences, regressors, x_features, feature_names, feature_subset, plot_interpretations):
    n_extreme = 3 # get the n_extreme best and worst
    sentences = sentences.loc[true.index]
    sentences['Gold Standard'] = true
    sentences['Predicted'] = predicted
    sentences['Diff'] = abs(sentences['Predicted'] - sentences['Gold Standard'])
    sentences.sort_values(by = 'Diff', inplace = True)
    best = sentences['Diff'].iloc[:n_extreme]
    print('Best Sentences Predicted:')
    for idx in best.index:
        print_sentence(sentences.loc[idx], idx)

    worst = sentences['Diff'].iloc[-n_extreme:]
    print('Worst Sentences Predicted:')
    for idx in worst.index:
        print_sentence(sentences.loc[idx], idx)

    sentences['Absolute Differences'] = abs(sentences['Predicted'] - sentences['Gold Standard'])
    sentences['Differences'] = sentences['Predicted'] - sentences['Gold Standard']
    print('Mean Differences for each Origin')
    print(sentences[['Origin', 'Absolute Differences', 'Differences']].groupby('Origin').mean())
    
    if not plot_interpretations:
        return

    best_stored = get_interpretations(best, sentences,  true, regressors, x_features, feature_names, feature_subset)
    worst_stored = get_interpretations(worst, sentences, true, regressors, x_features, feature_names, feature_subset)

    interpretations_plot(best_stored, 'best', feature_names)
    interpretations_plot(worst_stored, 'worst', feature_names)

