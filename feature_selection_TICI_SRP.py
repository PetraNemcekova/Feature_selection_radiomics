import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from mrmr import mrmr_classif
from skfda.preprocessing.dim_reduction import variable_selection
from skfda.representation.grid import FDataGrid
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from statsmodels.formula.api import ols


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

class Preprocessing:

    # def Initialize_file_with_results_patients(patient_info, path_to_output_file):
    #     patients_for_analysis = Preprocessing.Selection_of_balanced_dataset_of_TICIs_binary(patient_tici_info = patient_info, label_of_patient_ID = patient_info.columns[0], label_of_tici = patient_info.columns[1])
    #     file_with_results = pd.DataFrame(data = {'patient_ID' : patients_for_analysis.iloc[:,0], 'TICI': patients_for_analysis.iloc[:,1]})
    #     file_with_results.to_csv(path_to_output_file, index= False)
    #     return(file_with_results['patient_ID'])
    

    def Selection_of_balanced_dataset_of_TICIs_binary(patient_tici_info, label_of_tici):

        ticis_labels = np.unique(patient_tici_info[label_of_tici].tolist())[:-1]
        count_of_patients_with_assigned_tici = np.zeros((2,len(ticis_labels)), dtype = object)
        for index, tici_label in enumerate(ticis_labels):
            count_of_patients_with_assigned_tici[0, index] = tici_label
            count_of_patients_with_assigned_tici[1, index] = patient_tici_info[label_of_tici].tolist().count(tici_label)

        number_of_patients_for_analysis_one_half = np.round(np.sum(count_of_patients_with_assigned_tici[1,0:4])*4/5)
        patients_without_succesfull_reperfusion = patient_tici_info.loc[patient_tici_info[label_of_tici].isin(ticis_labels[0:4])]
        patients_with_succesfull_reperfusion = patient_tici_info.loc[patient_tici_info[label_of_tici].isin(ticis_labels[4:])]
        selected_patients_without_succesfull_reperfusion = patients_without_succesfull_reperfusion.reindex(np.random.permutation(patients_without_succesfull_reperfusion.index))[0:int(number_of_patients_for_analysis_one_half)]
        selected_patients_with_succesfull_reperfusion = patients_with_succesfull_reperfusion.reindex(np.random.permutation(patients_with_succesfull_reperfusion.index))[0:int(number_of_patients_for_analysis_one_half)]

        patients_for_analysis = pd.concat([selected_patients_with_succesfull_reperfusion, selected_patients_without_succesfull_reperfusion])
        patients_for_analysis.to_csv('Balanced_dataset_Successfull_recanalisation.csv', index= False)
        return(patients_for_analysis)
    
    def remove_features_NaN(all_features_all_patients_df):

        all_patients_all_features_without_nans_df = all_features_all_patients_df.dropna(axis = 1)

        return(all_patients_all_features_without_nans_df)
       
class Voxel_preprocessing_one_acquisition():


    def Normalisation(data_with_metadata_df_to_preprocess, normalisation_method, label_to_save, normalise_through):

        metadata = data_with_metadata_df_to_preprocess[['patient_ID', 'acquisition', 'vx_position']].copy()
        data_to_preprocess = data_with_metadata_df_to_preprocess.drop(['Unnamed: 0','patient_ID', 'acquisition', 'vx_position'], axis = 1)

        if normalise_through == 'Patient':
            patients = np.unique(metadata['patient_ID'])
            data_normalised_for_analysis = pd.DataFrame()
            data_normalised_for_analysis = pd.concat([data_normalised_for_analysis, metadata])
            for patient in patients:
                patients_features = data_to_preprocess.loc[metadata['patient_ID'] == patient]
                if (normalisation_method== 'min_max'):
                    data_normalised_for_analysis = pd.concat([data_normalised_for_analysis, (patients_features - patients_features.min())/(patients_features.max()-patients_features.min())])
                elif (normalisation_method == 'z_score'):
                    data_normalised_for_analysis =  pd.concat([data_normalised_for_analysis,(patients_features - patients_features.mean())/patients_features.std()])
        
        elif  normalise_through == 'Features':
            if (normalisation_method== 'min_max'):
                data_normalised_for_analysis = (data_to_preprocess - data_to_preprocess.min())/(data_to_preprocess.max()-data_to_preprocess.min())
            elif (normalisation_method == 'z_score'):
                data_normalised_for_analysis = (data_to_preprocess - data_to_preprocess.mean())/data_to_preprocess.std()
            data_normalised_for_analysis = pd.concat([data_normalised_for_analysis, metadata], axis = 1)

        else:
            print('No possibility to normalise through:' + normalise_through)

        data_normalised_for_analysis.columns = [feature_name.replace('-', '_') for feature_name in data_normalised_for_analysis.columns.to_list()]
        data_normalised_for_analysis.columns = [feature_name.replace('.nrrd', '') for feature_name in data_normalised_for_analysis.columns.to_list()]
        data_normalised_for_analysis.to_csv(label_to_save + '_' + normalisation_method + '_normalised_through_' + normalise_through + '.csv', index= False)

        return data_normalised_for_analysis


    def Create_balanced_datasets_for_analysis(dataset, acquisition, information_about_balanced_dataset, path_to_csv_files):
        balanced_features, balanced_metadata = Voxel_preprocessing_one_acquisition.Create_balanced_dataset_according_to_Successful_Reperfusion(dataset = dataset, 
                                                                                                                                               acquisition = acquisition, 
                                                                                                                                               information_about_balanced_dataset = information_about_balanced_dataset, 
                                                                                                                                               path_to_csv_files = path_to_csv_files)
        balanced_metadata_voxels, balanced_features_voxels = Voxel_preprocessing_one_acquisition.Create_balanced_dataset_according_to_SRP_and_number_of_voxels(balanced_features = balanced_features, 
                                                                                                          balanced_metadata = balanced_metadata, 
                                                                                                          acquisition = acquisition, 
                                                                                                          path_to_csv_files = path_to_csv_files)
        return(balanced_metadata_voxels, balanced_features_voxels)

    def Create_balanced_dataset_according_to_Successful_Reperfusion(dataset, acquisition, information_about_balanced_dataset, path_to_csv_files):
        select_rows_for_balanced_dataset = ((dataset['acquisition'] == acquisition) & (dataset['patient_ID'].isin(information_about_balanced_dataset['studysubjectid']))).tolist()

        balanced_metadata = dataset.loc[select_rows_for_balanced_dataset, ['patient_ID', 'vx_position', 'acquisition']].copy()
        dataset = dataset.drop( ['patient_ID', 'vx_position', 'acquisition'], axis = 1)
        balanced_features = dataset.loc[select_rows_for_balanced_dataset, :]
        balanced_metadata.to_csv(path_to_csv_files + 'balanced_dataset_metadata_noiv_' + acquisition + '.csv', index= False)
        balanced_features.to_csv(path_to_csv_files + 'balanced_dataset_features_noiv_' + acquisition + '.csv', index= False)
        return(balanced_features, balanced_metadata)

    def Create_balanced_dataset_according_to_SRP_and_number_of_voxels(balanced_features, balanced_metadata, acquisition, path_to_csv_files):
        selected_voxels_indices = []
        number_of_voxels_per_patient = np.min(balanced_metadata['patient_ID'].value_counts())
        patients = np.unique(balanced_metadata['patient_ID'])
        for patient in patients:
            actual_patient_selected_voxels = balanced_metadata.loc[balanced_metadata['patient_ID'] == patient].sample(n = number_of_voxels_per_patient)
            selected_voxels_indices.extend(actual_patient_selected_voxels.index.to_list())
        balanced_metadata_voxels = balanced_metadata[balanced_metadata.index.isin(selected_voxels_indices)]
        balanced_features_voxels = balanced_features[balanced_features.index.isin(selected_voxels_indices)]
        balanced_metadata_voxels.to_csv(path_to_csv_files + 'balanced_dataset_voxels_metadata_noiv_' + acquisition + '.csv', index= False)
        balanced_features_voxels.to_csv(path_to_csv_files + 'balanced_dataset_voxels_features_noiv_' + acquisition + '.csv', index= False)

        return(balanced_metadata_voxels, balanced_features_voxels)
    
class Feature_Selection:
    def get_features_by_correlations(all_features_for_correlation_analysis, upper_threshold, acquisition_type, output_path):
            features_for_correlation_analysis = all_features_for_correlation_analysis.copy()
            corr_matrix = features_for_correlation_analysis.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > upper_threshold)]
            features_for_correlation_analysis.drop(to_drop, axis=1, inplace=True)
            Basics.Save_List_To_Txt(list_to_save = features_for_correlation_analysis.columns.to_list(), output_file_name = output_path + 'FeaturesAfterCorrelationAnalysisThr' + str(upper_threshold) + '_' + acquisition_type + '.txt')
            return features_for_correlation_analysis.columns
    
    def select_features_by_LDA(training_data, training_labels, testing_data):
        lda_model = LinearDiscriminantAnalysis(solver = 'eigen', store_covariance = True)
        lda_model.fit(training_data, training_labels)
        predicted_labels = lda_model.predict(testing_data)
        feature_weights = lda_model.coef_
        feature_weights_sorted = feature_weights[0][np.abs(feature_weights[0]).argsort()[::-1]]
        features_sorted = [training_data.columns.to_list()[i] for i in list(np.abs(feature_weights[0]).argsort()[::-1])]

        return(predicted_labels, feature_weights_sorted, features_sorted)
    
    def select_features_by_ANOVA(training_data, training_labels):

        formula = 'label ~ ' + ' + '.join(training_data.columns)
        learning_model = ols(formula = formula, data = pd.concat([training_data.reset_index(drop = True), pd.DataFrame({'label' : training_labels})], axis = 1)).fit()
        table = sm.stats.anova_lm(learning_model)
        # features_statisticaly_significant = training_data.columns[np.where(table['PR(>F)']<0.05)]
        sorted_features = table.sort_values(by = 'PR(>F)')['PR(>F)']

        return(sorted_features)
    
    def select_features_by_MRMR(training_data, training_labels):

        features_sorted = mrmr_classif(X = training_data, y = training_labels, K = len(training_data.columns))

        return(features_sorted)
    
    
    def do_cross_validation(metadata, features, info_to_balanced_dataset, threshold, acquisition):
        ticis_labels = np.unique(info_to_balanced_dataset[info_to_balanced_dataset.columns[1]].tolist())

        SRP_patients = info_to_balanced_dataset.loc[info_to_balanced_dataset[ticis.columns[1]].isin(ticis_labels[0:4])][ticis.columns[0]].to_list()
        nSRP_patients = info_to_balanced_dataset.loc[info_to_balanced_dataset[ticis.columns[1]].isin(ticis_labels[4:])][ticis.columns[0]].to_list()
        fold_size = len(nSRP_patients) // 5

        result_table_ANOVA = pd.DataFrame()
        result_table_LDA = pd.DataFrame()
        result_table_MRMR = pd.DataFrame()
        accuracies_LDA = pd.DataFrame()
        iteration = 1
        column_labels = []

        for fold in range(0, len(nSRP_patients) - fold_size, fold_size):
            column_label = 'cv_fold_' + str(iteration)
            column_labels.append(column_label)
            SRP_patients_ids_testing = SRP_patients[fold:fold + fold_size]
            nSRP_patients_ids_testing = nSRP_patients[fold:fold + fold_size]

            SRP_patients_ids_training = list(set(SRP_patients) - set(SRP_patients_ids_testing))
            nSRP_patients_ids_training = list(set(nSRP_patients) - set(nSRP_patients_ids_testing))

            SRP_patients_features_testing = features[(metadata['patient_ID'].isin(SRP_patients_ids_testing)).reset_index(drop = True)]
            nSRP_patients_features_testing = features[(metadata['patient_ID'].isin(nSRP_patients_ids_testing)).reset_index(drop = True)]

            SRP_patients_features_training = features[(metadata['patient_ID'].isin(SRP_patients_ids_training)).reset_index(drop = True)]
            nSRP_patients_features_training = features[(metadata['patient_ID'].isin(nSRP_patients_ids_training)).reset_index(drop = True)]

            training_set_features = pd.concat([SRP_patients_features_training, nSRP_patients_features_training])
            testing_set_features = pd.concat([SRP_patients_features_testing, nSRP_patients_features_testing])

            training_set_labels = np.append(np.ones(len(SRP_patients_features_training)), np.zeros(len(nSRP_patients_features_training)))
            testing_set_labels = np.append(np.ones(len(SRP_patients_features_testing)), np.zeros(len(nSRP_patients_features_testing)))

            testing_set_predicted_labels_LDA, feature_weights_sorted_LDA, features_sorted_LDA = Feature_Selection.select_features_by_LDA(training_data = training_set_features, training_labels = training_set_labels, testing_data = testing_set_features)
            accuracy_LDA = accuracy_score(testing_set_labels, testing_set_predicted_labels_LDA)
            features_sorted_ANOVA = Feature_Selection.select_features_by_ANOVA(training_data = training_set_features, training_labels = training_set_labels)
            features_sorted_MRMR = Feature_Selection.select_features_by_MRMR(training_data = training_set_features, training_labels = training_set_labels)

            result_table_ANOVA = pd.concat([result_table_ANOVA, features_sorted_ANOVA],axis = 1)
            result_table_MRMR[column_label] = features_sorted_MRMR
            result_table_LDA[column_label + '_features_sorted'] = features_sorted_LDA 
            result_table_LDA[column_label + '_weigths'] = feature_weights_sorted_LDA 

            accuracies_LDA['acc_' + column_label] = [accuracy_LDA]
            iteration += 1

        result_table_ANOVA.columns = column_labels
        result_table_ANOVA.to_csv(acquisition + '_ANOVA_results_' + str(threshold) + '.csv')
        result_table_MRMR.to_csv(acquisition + '_MRMR_results_' + str(threshold) + '.csv')
        result_table_LDA.to_csv(acquisition + '_LDA_results_' + str(threshold) + '.csv')
        accuracies_LDA.to_csv(acquisition + '_LDA_accuracies_' + str(threshold) + '.csv')
        return()

class Dimensional_space:
    def get_principle_components_by_PCA(data_for_PCA_df, metadata, model, database, output_path, threshold_PCs_explained_variability, threshold_feature_explained_variability):

        pca_model = PCA()
        transformed_data_PCA = pca_model.fit_transform(data_for_PCA_df)
        explained_variance_ratio_PCA = pca_model.explained_variance_ratio_
        explained_variance_ratio_cumulative = np.cumsum(explained_variance_ratio_PCA)
        size_of_int = len(str(np.shape(data_for_PCA_df)[1]))
        PCs_names = [f'PC_' + str(i).zfill(size_of_int) for i in range(1,np.shape(data_for_PCA_df)[1]+1)]
        Dimensional_space.PCA_explained_variance_visualisation(explained_variance_PCs = explained_variance_ratio_cumulative, output_path = output_path + 'PCA_between_patients\\', model = model, database = database, threshold = threshold_PCs_explained_variability)
        selected_PCs, index_of_last_needed_PC = Dimensional_space.selection_of_the_PCs_with_explained_variance(output_of_PCA_arr = transformed_data_PCA, explained_variance_ratio_cumulative = explained_variance_ratio_cumulative, explained_variance_threshold = threshold_PCs_explained_variability)
        explained_variance_features = pd.DataFrame(data = pca_model.components_, columns = data_for_PCA_df.columns, index = PCs_names).iloc[:index_of_last_needed_PC, :]
        features_for_further_analysis = Dimensional_space.selection_of_most_important_features(explained_variance_PCs_df = explained_variance_ratio_PCA, explained_variance_features_df = explained_variance_features, model = model, threshold = threshold_feature_explained_variability, output_path = output_path )
        Dimensional_space.PCA_postprocessing(output_of_PCA_arr = selected_PCs, metadata = metadata, label_to_save = 'PCs_normalised_min_max_', model = model, output_path = output_path)

        return features_for_further_analysis
    

    def PCA_postprocessing(output_of_PCA_arr, metadata, label_to_save, model, output_path):
        size_of_int = len(str(np.shape(output_of_PCA_arr)[1]))
        column_names = [f'PC_' + str(i).zfill(size_of_int) for i in range(1,np.shape(output_of_PCA_arr)[1]+1)]
        output_of_PCA_df = pd.DataFrame(output_of_PCA_arr, columns = column_names)
        output_of_PCA_df['patient_ID'] = metadata['patient_ID']
        output_of_PCA_df['vx_position'] = metadata['vx_position']

        output_of_PCA_df.to_csv(output_path + label_to_save + model + '.csv', index= False)
        return output_of_PCA_df
    
    def selection_of_the_PCs_with_explained_variance(output_of_PCA_arr, explained_variance_ratio_cumulative, explained_variance_threshold):

        last_needed_PC_index = np.where(explained_variance_ratio_cumulative > explained_variance_threshold)[0][0]
        PCs_with_most_explained_variance = output_of_PCA_arr[:,0:last_needed_PC_index]

        return(PCs_with_most_explained_variance, last_needed_PC_index)
    
    def PCA_explained_variance_visualisation(explained_variance_PCs, output_path, model, database, threshold):

        plt.figure()
        plt.plot(explained_variance_PCs)
        plt.xlabel('Components')
        plt.ylabel('Explained variance')
        plt.axhline(y = threshold, color = 'red', linestyle = '--')
        plt.show()
        plt.savefig(output_path + 'Explained_variance_cumulative_' + model + '_' + database + '.png')
        plt.clf()
        plt.close()

    def selection_of_most_important_features(explained_variance_PCs_df, explained_variance_features_df, model, threshold, output_path):

        pivoted_table_for_visualisation = Dimensional_space.PCA_preprocessing_for_feature_importance_visualisation(explained_variance_features_df = explained_variance_features_df)
        Dimensional_space.Visualise_Heatmap_of_feature_variances(table_for_visualisation = pivoted_table_for_visualisation, model = model, output_path = output_path)
        
        weighted_sums_of_feature_importances = (explained_variance_features_df * explained_variance_PCs_df[:explained_variance_features_df.shape[0],np.newaxis]).sum().abs().sort_values(ascending = False)
        selected_features = weighted_sums_of_feature_importances.index[np.where(weighted_sums_of_feature_importances > threshold)].tolist()
        Basics.Save_List_To_Txt(output_file_name = output_path + 'features_for_further_analysis.txt', list_to_save = selected_features)

        return selected_features

    def PCA_preprocessing_for_feature_importance_visualisation(explained_variance_features_df):

        vector_of_features = []
        vector_of_PCs = []
        vector_of_values = np.empty(0)

        features = explained_variance_features_df.columns.to_list()

        for index,row in explained_variance_features_df.iterrows():
            current_PC = [index] * len(features)
            values = np.array(row[features])
            vector_of_PCs = vector_of_PCs + current_PC
            vector_of_features = vector_of_features + features
            vector_of_values = np.concatenate((vector_of_values, values))
        
        table_for_visualisation = pd.DataFrame({'Components': vector_of_PCs, 'Features': vector_of_features, 'Value': vector_of_values.astype(float)}).pivot(index = 'Features', columns = 'Components', values = 'Value')
        
        return(table_for_visualisation)
    
    def Visualise_Heatmap_of_feature_variances(table_for_visualisation, output_path, model):
        
        plt.figure(figsize = (20, 8))
        sns.heatmap(table_for_visualisation, fmt='.5f', cmap = 'Spectral')
        plt.tight_layout()
        plt.show()
        plt.savefig(output_path + model + '_visualisation_feature_variances_PC_heatmap.png')
        plt.clf()
        plt.close()


        return()

if __name__ == "__main__":
    path_to_maps_noiv_resampled = 'L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\NOIV\\Last\\F_maps_Resampled_05\\'
    path_to_maps_noiv_not_resampled = 'L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\NOIV\\Last\\F_maps_Without_resampling\\'
    path_to_thr_masks_noiv = 'L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\NOIV\\Last\\Cropped_Images_Resampled_05\\'
    path_to_tici_noiv = 'L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\noiv_clinical_info.csv'    
    path_to_tici_registry = 'L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\registry_clinical_info.csv' 
    path_to_clustered_data = 'L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\NOIV\\Last\\Final_Final\\'
    path_to_3D_UMAP_maps = 'L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\NOIV\\Last\\Clustering_final\\UMAP_maps\\last\\'
    path_to_csvs = 'L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\NOIV\\Last\\Final_Final\\Csvs\\'
    path_to_images = 'L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\NOIV\\Last\\Final_Final\\Images\\'
    path_to_feature_maps = 'L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\NOIV\Last\\F_maps_Resampled_05\\'
    path_to_txts_of_selected_features = 'L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\NOIV\\Last\\Feature_selection\\'

    thresholds = [0.6, 0.5, 0.4, 0.3]
    output_file_name_stack = 'concatenated_all_info_noiv_050505'
    acquisitions = ['ncct', 'cta']
    database = 'noiv'
    
    if database == 'noiv':
        ticis = pd.read_csv("L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\noiv_clinical_info.csv", usecols=['studysubjectid','post_etici'])
        tici_values = np.unique(ticis['post_etici'].dropna(axis = 0))

    elif database == 'registry':
        ticis = pd.read_csv("L:\\basic\\divi\\CMAJOIE\\projects\\mahsa_petra\\registry_clinical_info.csv", usecols=['StudySubjectID','posttici_c'])
        tici_values = np.unique(ticis['posttici_c'].dropna(axis = 0))

    assigned_numbers = np.arange(0,len(tici_values))
    tici_dictionary  = dict(zip(tici_values, assigned_numbers))


    heterogeneity_map_types = np.unique(['_'.join(filename.split('_')[2:]) for filename in os.listdir(
        path_to_maps_noiv_not_resampled + '10002\\') if filename.endswith('.nrrd')]).tolist()
    
    
    output_file_name_stack = 'concatenated_all_info_noiv_050505'
    database_features_all = pd.read_csv(path_to_maps_noiv_resampled +  output_file_name_stack + '.csv' ) # values already without bcg
    noiv_database_features_without_NaN = Preprocessing.remove_features_NaN(database_features_all)
    # balanced_dataset_info = Preprocessing.Selection_of_balanced_dataset_of_TICIs_binary(patient_tici_info= ticis, label_of_tici=ticis.columns[1])
    balanced_dataset_info = pd.read_csv('Balanced_dataset_Successfull_recanalisation.csv')
    # balanced_dataset_info.columns = [feature_name.replace('-', '_') for feature_name in balanced_dataset_info.columns.to_list()]
    # balanced_dataset_info.columns = [feature_name.replace('.nrrd', '') for feature_name in balanced_dataset_info.columns.to_list()]

    for acquisition in acquisitions: 

        all_features_current_acq = noiv_database_features_without_NaN.loc[noiv_database_features_without_NaN['acquisition']==acquisition]
        norm_features_all_patients_acq = Voxel_preprocessing_one_acquisition.Normalisation(data_with_metadata_df_to_preprocess = all_features_current_acq, 
                                                                                           normalisation_method = 'min_max', 
                                                                                           label_to_save = 'balanced_dataset_features_noiv_' + acquisition, 
                                                                                           normalise_through = 'Features')
        # norm_features_all_patients_acq = pd.read_csv('balanced_dataset_features_noiv_' + acquisition + '_min_max_normalised_through_Features.csv')
        metadata_for_selection, features_for_selection = Voxel_preprocessing_one_acquisition.Create_balanced_datasets_for_analysis(dataset = norm_features_all_patients_acq, 
                                                                                  acquisition = acquisition, 
                                                                                  information_about_balanced_dataset = balanced_dataset_info, 
                                                                                  path_to_csv_files = path_to_csvs)
        # metadata_for_selection = pd.read_csv(r'L:\basic\divi\CMAJOIE\projects\mahsa_petra\NOIV\Last\Final_Final\Csvs\balanced_dataset_voxels_metadata_noiv_ncct.csv')
        # features_for_selection = pd.read_csv(r'L:\basic\divi\CMAJOIE\projects\mahsa_petra\NOIV\Last\Final_Final\Csvs\balanced_dataset_voxels_features_noiv_ncct.csv').drop('Unnamed: 0', axis = 1)

        # features_for_selection.columns = [feature_name.replace('-', '_') for feature_name in features_for_selection.columns.to_list()]
        # features_for_selection.columns = [feature_name.replace('.nrrd', '') for feature_name in features_for_selection.columns.to_list()]

        for threshold in thresholds:
            
            # Correlation analysis selection
            features_selected_by_correlation_acq = Feature_Selection.get_features_by_correlations(all_features_for_correlation_analysis = norm_features_all_patients_acq.loc[:, ~norm_features_all_patients_acq.columns.isin(['patient_ID', 'acquisition', 'vx_position'])], 
                                                                                              upper_threshold = threshold, 
                                                                                              acquisition_type = acquisition + '_' + database, 
                                                                                              output_path = path_to_txts_of_selected_features)

            features_selected_by_correlation_acq = Basics.Read_Feature_Names_Txt_To_List(path_to_txts_of_selected_features + 'FeaturesAfterCorrelationAnalysisThr' + str(threshold) + '_' + acquisition + '_' + database + '.txt')[:-1]
            # features_selected_by_correlation_acq = [feature_name.replace('-', '_') for feature_name in features_selected_by_correlation_acq]            
            # features_selected_by_correlation_acq = [feature_name.replace('.nrrd', '') for feature_name in features_selected_by_correlation_acq]
            features_for_selection_after_cc = pd.read_csv(path_to_csvs + 'balanced_dataset_voxels_features_noiv_' + acquisition + '.csv', 
                                                          usecols=features_selected_by_correlation_acq)
            
            # PCA selection
            features_for_selection_after_PCA = Dimensional_space.get_principle_components_by_PCA(data_for_PCA_df = features_for_selection_after_cc, 
                                                                                                                              metadata = metadata_for_selection, 
                                                                                                                              model = acquisition + '_cc_' + str(threshold), 
                                                                                                                              database = database, 
                                                                                                                              output_path = path_to_images + 'balanced_PCA\\', 
                                                                                                                              threshold_PCs_explained_variability = 0.9, 
                                                                                                                              threshold_feature_explained_variability = 0.02)
            feature_labels_for_selection_after_PCA =  Basics.Read_Feature_Names_Txt_To_List( path_to_images + 'balanced_PCA\\' + 'features_for_further_analysis.txt')[:-1]
            features_for_selection_after_PCA = pd.read_csv(path_to_csvs + 'balanced_dataset_voxels_features_noiv_' + acquisition + '.csv', 
                                                          usecols=feature_labels_for_selection_after_PCA)
            # Cross validation - preparation of folds
            Feature_Selection.do_cross_validation(metadata = metadata_for_selection, 
                                                  features = features_for_selection_after_PCA, 
                                                  info_to_balanced_dataset = balanced_dataset_info, 
                                                  threshold = threshold, 
                                                  acquisition = acquisition)


            # ANOVA selection

            # LDA selection

            # MRMR selection

    
    # # features_after_correlation = Feature_Selection.get_features_by_correlations(all_features_for_correlation_analysis = , upper_threshold = , acquisition_type)
    # # balanced_dataset_info = Preprocessing.Selection_of_balanced_dataset_of_TICIs_binary(patient_tici_info= ticis, label_of_patient_ID= ticis.columns[0], label_of_tici=ticis.columns[1])
    # balanced_dataset_info = pd.read_csv('Balanced_dataset_Successfull_recanalisation.csv')
    # # patients = os.listdir(path_to_maps_noiv_resampled)

    # for acquisition in acquisitions:
    #     for threshold in thresholds:
    #         for metric in metrics:
    #             # threshold = 0.6
    #             model = metric + '_' + str(threshold) + '_' + acquisition
    #             # noiv_fs_using_cc_acq = Basics.Read_Feature_Names_Txt_To_List('FeaturesAfterCorrelationAnalysisThr' + str(threshold) + acquisition + '_noiv.txt')[:-1]
    #             # noiv_features_by_correlation_analysis_data_acq = pd.read_csv(path_to_maps_noiv_resampled + 'normalised_features_' + database + '_' + acquisition +'.csv', usecols=noiv_fs_using_cc_acq)
    
    #             # noiv_database_features_acquisition_metadata = pd.read_csv(output_file_name_stack + '.csv', usecols=['patient_ID', 'vx_position', 'acquisition'])
    #             # noiv_database_features_acquisition = pd.read_csv(output_file_name_stack + '.csv', usecols=noiv_fs_using_cc_acq)

    #             # select_rows_for_balanced_dataset = ((noiv_database_features_acquisition_metadata['acquisition'] == acquisition) & (noiv_database_features_acquisition_metadata['patient_ID'].isin(balanced_dataset_info['studysubjectid']))).tolist()
    #             # balanced_metadata = noiv_database_features_acquisition_metadata.loc[select_rows_for_balanced_dataset]
    #             # balanced_features = noiv_database_features_acquisition.loc[select_rows_for_balanced_dataset]

    #             # balanced_metadata.to_csv('balanced_dataset_metadata_noiv_' + model + '.csv', index= False)
    #             # balanced_features.to_csv('balanced_dataset_features_noiv_' + model + '.csv', index= False)
    #             balanced_metadata = pd.read_csv('balanced_dataset_metadata_noiv_' + model + '.csv')
    #             balanced_features = pd.read_csv('balanced_dataset_features_noiv_' + model + '.csv')

    #             patients = np.unique(balanced_metadata['patient_ID'].tolist())
                