import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from scipy.stats import spearmanr

class Basics:
    def Save_List_To_Txt(list_to_save, output_file_name):

        with open(output_file_name, 'w') as fp:
            for item in list_to_save :
                # write each item on a new line
                fp.write("%s\n" % item)

        return()
    
    
    def Read_Feature_Names_Txt_To_List(file_name):

        with open(file_name, 'r') as file_to_read:

            data = file_to_read.read() 
            data_into_list = data.split("\n") 

        return(data_into_list)
    
class Analysis:
    def Correlation_analysis_between_acquisitions(ncct_features, cta_features):
        weights_ncct = ncct_features.loc['sum']
        weights_cta = cta_features.loc['sum']
        orders_ncct = rankdata(-weights_ncct, method='dense').astype(int)
        orders_cta = rankdata(-weights_cta, method='dense').astype(int)

        correlation_result = spearmanr(orders_ncct, orders_cta)
        return(correlation_result.pvalue, correlation_result.statistic)

class Visualisation_of_results:
    def Feature_importance_visualisation(path_to_table_with_results_ncct, path_to_table_with_results_cta, output_path):

        table_with_results_ncct = pd.read_csv(path_to_table_with_results_ncct)
        table_with_results_cta = pd.read_csv(path_to_table_with_results_cta)
        table_with_results_ncct_unified, table_with_results_cta_unified = Visualisation_of_results.Unify_features_between_acquisitions(table_with_results_ncct = table_with_results_ncct, 
                                                                                                                                       table_with_results_cta = table_with_results_cta)

        heatmap_data_normalised_cta = Visualisation_of_results.Preprocessing_for_heatmap_visualisation(table_with_results = table_with_results_cta_unified, sort_by = 'sum')
        heatmap_data_normalised_ncct = Visualisation_of_results.Preprocessing_for_heatmap_visualisation(table_with_results = table_with_results_ncct_unified, sort_by = heatmap_data_normalised_cta.columns.to_list() )
        # heatmap_data_normalised_ncct = Visualisation_of_results.Preprocessing_for_heatmap_visualisation(table_with_results = table_with_results_ncct_unified, sort_by = 'sum' )

        plt.figure(figsize = (8,4))
        plt.tight_layout()
        plt.subplot(2,1,1)

        sns.heatmap(heatmap_data_normalised_cta, fmt='.5f', cmap = 'Spectral_r', vmax = 1, vmin = 0, xticklabels=False)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.ylabel("CTA")
        plt.xlabel('')
        plt.subplot(2,1,2)

        sns.heatmap(heatmap_data_normalised_ncct, fmt='.5f', cmap = 'Spectral_r', vmax = 1, vmin = 0, xticklabels=True)
        plt.tight_layout()
        plt.yticks(rotation=45)
        plt.ylabel("NCCT")
        plt.xlabel('')
        plt.show()
        plt.savefig(output_path + 'Visualisation_feature_variances_PC_heatmap_same_order.png')
        plt.clf()
        plt.close()
       
        return(heatmap_data_normalised_ncct, heatmap_data_normalised_cta)

    def Preprocessing_for_heatmap_visualisation(table_with_results, sort_by = 'sum'):
        
        df_melted = table_with_results.melt(id_vars=['Feature_ID'], value_vars=['ANOVA', 'MRMR', 'LDA', 'sum'], 
                    var_name='Metric', value_name='Value')
        heatmap_data = df_melted.pivot(index='Metric', columns='Feature_ID', values='Value')
        heatmap_data_normalised = heatmap_data.loc[['ANOVA', 'LDA', 'MRMR']]/20
        heatmap_data_normalised.loc['sum'] = heatmap_data.loc['sum']/60
        if sort_by == 'sum':
            heatmap_data_normalised = heatmap_data_normalised.sort_values(by = sort_by, axis = 1, ascending = False)
        else:
            heatmap_data_normalised = heatmap_data_normalised[sort_by]

        return(heatmap_data_normalised)
    
    def add_missing_features(all_features, data):

        missing_features = [feature for feature in all_features if feature not in data['Feature_name'].values]
        for feature in missing_features:
            data = pd.concat([data, pd.DataFrame({'Feature_name': [feature], 'ANOVA': [0], 'MRMR': [0], 'LDA': [0], 'sum': [0]})])
        return(data)

    def Unify_features_between_acquisitions(table_with_results_ncct, table_with_results_cta):

        features = table_with_results_cta['Feature_name'].to_list()
        features.extend(table_with_results_ncct['Feature_name'].to_list())
        unique_features = list(set(features))
        feature_ids = np.linspace(1, len(unique_features), len(unique_features)).astype(int)
        feature_id_dict = dict(zip(unique_features, feature_ids))

        table_with_results_ncct = Visualisation_of_results.add_missing_features(all_features = unique_features, data = table_with_results_ncct)
        table_with_results_cta = Visualisation_of_results.add_missing_features(all_features = unique_features, data = table_with_results_cta)

        # Map feature names in the 'Feature_name' column to feature IDs
        table_with_results_ncct['Feature_ID'] = table_with_results_ncct['Feature_name'].map(feature_id_dict).fillna(0).astype(int)
        table_with_results_cta['Feature_ID'] = table_with_results_cta['Feature_name'].map(feature_id_dict).fillna(0).astype(int)
        return(table_with_results_ncct, table_with_results_cta)


path_to_results = 'E:\\Projects\\Stroke\\Feature_selection_results\\Results_selection_share\\'
acquisitions = ['ncct', 'cta']
thresholds = ['0.6','0.5','0.4','0.3']
folds = ['cv_fold_1', 'cv_fold_2', 'cv_fold_3', 'cv_fold_4', 'cv_fold_5']
methods = ['ANOVA', 'MRMR', 'LDA']

for acquisition in acquisitions:

    result_table_acquisition_all_thresholds = pd.DataFrame(columns= ['Feature_name', 'ANOVA', 'MRMR', 'LDA'])
    feature_names = []
    all_features_names_acquisition = list(set([
    feature 
    for threshold in thresholds 
    for feature in pd.read_csv(
        path_to_results + '_'.join([acquisition, 'ANOVA', 'results', threshold]) + '.csv', 
        usecols=['Unnamed: 0']
    )['Unnamed: 0'].tolist()
    ]))
    result_table_acquisition_all_thresholds['Feature_name'] = all_features_names_acquisition
    result_table_acquisition_all_thresholds[['ANOVA', 'MRMR', 'LDA', 'sum']] = 0


    for threshold in thresholds:


        ANOVA_results_threshold = pd.read_csv('_'.join([acquisition, 'ANOVA', 'results', threshold]) + '.csv') # Unnamed: 0 cv_fold_1
        LDA_results_threshold = pd.read_csv('_'.join([acquisition, 'LDA', 'results', threshold]) + '.csv') # cv_fold_1_features_sorted cv_fold_1_weights
        MRMR_results_threshold = pd.read_csv('_'.join([acquisition, 'MRMR', 'results', threshold]) + '.csv') # cv_fold_1
        
        feature_names = list(ANOVA_results_threshold['Unnamed: 0'])
        result_table_acquisition_one_threshold = pd.DataFrame(columns= ['Feature_name', 'ANOVA', 'MRMR', 'LDA'])
        result_table_acquisition_one_threshold['Feature_name'] = feature_names
        result_table_acquisition_one_threshold[['ANOVA', 'MRMR', 'LDA']] = 0

        for fold in folds:
            ANOVA_features_statisticaly_significant = ANOVA_results_threshold.loc[ANOVA_results_threshold[fold] < 0.05]['Unnamed: 0'].to_list()
            result_table_acquisition_one_threshold.loc[result_table_acquisition_one_threshold['Feature_name'].isin(ANOVA_features_statisticaly_significant), 'ANOVA'] += 1
            LDA_important_features = LDA_results_threshold.loc[np.where(np.abs(LDA_results_threshold[fold + '_weigths']) >= np.max(np.abs(LDA_results_threshold[fold + '_weigths']))/3)][fold + '_features_sorted'].to_list()
            result_table_acquisition_one_threshold.loc[result_table_acquisition_one_threshold['Feature_name'].isin(LDA_important_features), 'LDA'] += 1
            MRMR_important_features = MRMR_results_threshold.iloc[0:int(np.mean([len(ANOVA_features_statisticaly_significant), len(LDA_important_features)]))]['cv_fold_1'].to_list()
            result_table_acquisition_one_threshold.loc[result_table_acquisition_one_threshold['Feature_name'].isin(MRMR_important_features), 'MRMR'] += 1
        
        result_table_acquisition_one_threshold.to_csv('summary_of_selected_features_through_folds_' + threshold + '_' + acquisition + '.csv', index= False)
        
        for method in methods:
            threshold_important_features = result_table_acquisition_one_threshold['Feature_name'].to_list()
            indices_in_df = [result_table_acquisition_all_thresholds.index[result_table_acquisition_all_thresholds['Feature_name'] == feature][0] for feature in threshold_important_features]            
            result_table_acquisition_all_thresholds.loc[indices_in_df, method] += result_table_acquisition_one_threshold[method].to_list()
        
        result_table_acquisition_all_thresholds['sum'] = result_table_acquisition_all_thresholds[['ANOVA', 'LDA', 'MRMR']].sum(axis = 'columns')
    result_table_acquisition_all_thresholds.to_csv('summary_of_selected_features_through_thresholds_' + acquisition + '.csv', index= False)
unified_ncct, unified_cta = Visualisation_of_results.Feature_importance_visualisation(path_to_table_with_results_ncct = 'summary_of_selected_features_through_thresholds_ncct.csv', 
                                                          path_to_table_with_results_cta = 'summary_of_selected_features_through_thresholds_cta.csv', 
                                                          output_path = path_to_results)
corr_p, corr_c = Analysis.Correlation_analysis_between_acquisitions(ncct_features = unified_ncct, 
                                                   cta_features = unified_cta)

print(f'P value: {corr_p}, cc: {corr_c}')

