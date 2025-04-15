import os 
import sys 
import argparse 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
sys.path.append('/project/kamoun_shared/code_shared/scikit-ExSTraCS/')
from skExSTraCS.ExSTraCS import ExSTraCS

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='full_data_path', help='path to target dataset ', type=str, default = 'mypath/myDataFolder/myDataset.txt') 
    parser.add_argument('--o', dest='outputPath', help='path to target output folder', type=str, default = 'mypath/myOutputFolder') 
    parser.add_argument('--ekf', dest='ekfolder', help='path to the folder containing ek scores for datasets', type=str, default = 'mypath/ekScores') 
    #Dataset Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome label (i.e. class label)', type=str, default = 'Class') 
    parser.add_argument('--il', dest='instanceID_label', help='label of instance ID column (if present)', type=str, default = 'InstanceID')
    parser.add_argument('--el', dest='excluded_column', help='label of another column to drop (if present)', type=str, default = 'Group') 
    #Critical Exstracs Parameters
    parser.add_argument('--it', dest='learning_iterations', help='number of rule training cycles', type=int, default=100000)
    parser.add_argument('--ps', dest='N', help='maximum micro rule population size', type=int, default=1000)
    parser.add_argument('--nu', dest='nu', help='power parameter', type=int, default=1)
    #Other Exstracs Parameters
    parser.add_argument('--ft', dest='do_attribute_tracking', help='feature tracking mechanism', type=str, default='True')
    parser.add_argument('--ff', dest='do_attribute_feedback', help='boolean flag to use feature tracking feedback', type=str, default='True')
    parser.add_argument('--c', dest='compaction', help='rule-compaction strategy', type=str, default='QRF')
    parser.add_argument('--ta', dest='track_accuracy_while_fit', help='boolean flag to use accuracy tracking', type=str, default='True')
    parser.add_argument('--sr', dest='stored_rule_iterations', help='comma-separated string indicating rule pop iterations to run full evaluation', type=str, default=None)
    parser.add_argument('--rs', dest='random_state', help='random state seed', type=int, default=42)
    #----------------------------------------------------------------------------------------------
    options=parser.parse_args(argv[1:])
    #Script Parameters
    full_data_path = options.full_data_path #full path to target dataset
    outputPath = options.outputPath
    ekfolder = options.ekfolder
    #Dataset Parameters
    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column
    #Critical Exstracs Parameters
    learning_iterations = options.learning_iterations
    N = options.N
    nu = options.nu
    #Other Exstracs Parameters
    if options.do_attribute_tracking == 'True':
        do_attribute_tracking = True
    else:
        do_attribute_tracking = False
    if options.do_attribute_feedback == 'True':
        do_attribute_feedback = True
    else:
        do_attribute_feedback = False
    compaction = options.compaction
    if options.track_accuracy_while_fit == 'True':
        track_accuracy_while_fit = True
    else:
        track_accuracy_while_fit = False

    if stored_rule_iterations is None or stored_rule_iterations == 'None':
        stored_rule_iterations = None
    else:
        stored_rule_iterations = options.stored_rule_iterations
    random_state = options.random_state

    # Get Dataset Name
    data_name = os.path.splitext(os.path.basename(full_data_path))[0]

    #Load/Process Training Dataset
    train_df = pd.read_csv(full_data_path, sep="\t")
    try:
        train_X = train_df.drop(excluded_column, axis=1) #Remove excluded column from consideration in this notebook
    except:
        train_X = train_df
        print('Excluded column not available')
    try:
        train_X = train_X.drop(instanceID_label,axis=1)
    except:
        print('Instance ID coulmn not available')
    print(train_X.columns)
    train_X = train_X.drop(outcome_label, axis=1)
    feature_names = train_X.columns # 6-bit multiplexer feature names are ['A_0','A_1','R_0', 'R_1', 'R_2','R_3']

    train_X = train_X.values
    train_y = train_df[outcome_label].values #outcome values
    try:
        row_id = train_df[instanceID_label].values #instance id values
    except:
        row_id = None

    #Load expert knowledge scores
    score_path_name = ekfolder+'/'+str(data_name)+'_MultiSURF_Scores.csv' #No need to change
    loaded_data = pd.read_csv(score_path_name)
    ek = loaded_data['Score'].tolist()

    #Train ExSTraCS --------------------------------------------------------------------------------------------------------------------
    exstracs = ExSTraCS(learning_iterations=learning_iterations,N=N,nu=nu,do_attribute_tracking=do_attribute_tracking,do_attribute_feedback=do_attribute_feedback,expert_knowledge=ek,rule_compaction=compaction,track_accuracy_while_fit=track_accuracy_while_fit,stored_rule_iterations=stored_rule_iterations,random_state=random_state)
        
    exstracs = exstracs.fit(train_X, train_y)

    # Save Rule Population
    exstracs.export_final_rule_population(feature_names,outcome_label,filename=outputPath+'/rule_pop_no_compact.csv',DCAL=True)
    exstracs.export_final_rule_population(feature_names,outcome_label,filename=outputPath+'/rule_pop.csv',DCAL=True,RCPopulation=True)
    #pop_df = exstracs.get_pop()
    #pop_df.to_csv(outputPath+'/rule_pop.csv', index=False)

    #Save Rule Population Performance Tracking Estimates
    exstracs.export_iteration_tracking_data(outputPath+'/rule_pop_tracking.csv')
    #tracking_df = heros.get_performance_tracking()
    #tracking_df.to_csv(outputPath+'/rule_pop_tracking.csv', index=False)

    #Save Plot Rule Pop Pareto Front
    #resolution = 500
    #plot_rules = True
    #color_rules = True
    #heros.get_rule_pareto_landscape(resolution, heros.rule_population, plot_rules, color_rules,show=True,save=True,output_path=outputPath)

    #Save Feature Tracking Scores
    ft = exstracs.get_attribute_tracking_scores(np.array(row_id))
    new_ft = []
    for i in range(len(ft)): #each row
        temp_list = []
        temp_list.append(ft[i][0]) #instanceID
        for j in range(len(ft[i][1])):
            temp_list.append(ft[i][1][j])
        new_ft.append(temp_list)
    header = ['InstanceID']
    for name in feature_names:
        header.append(name)

    ft_df = pd.DataFrame(new_ft,columns=header)

    #ft_df = pd.DataFrame(ft)
    #ft_df = heros.get_ft(feature_names)
    ft_df.to_csv(outputPath+'/feature_tracking_scores.csv', index=False)

    #Load/Process Testing Data
    test_data_path = full_data_path.replace("Train","Test")
    test_df = pd.read_csv(test_data_path, sep="\t")
    try:
        test_X = test_df.drop(excluded_column, axis=1)
    except:
        test_X = test_df
    try:
        test_X = test_X.drop(instanceID_label, axis=1)
    except:
        pass
    test_X = test_X.drop(outcome_label, axis=1)
    test_X = test_X.values
    test_y = test_df[outcome_label].values #outcome values

    #Convert population evaluation iterations from comma separated strings to lists
    if stored_rule_iterations == 'None' or stored_rule_iterations is None:
        stored_rule_iterations = None
    else:
        stored_rule_iterations = [int(value) for value in stored_rule_iterations.split(',')]

    #Initialize results dataframe information
    headers = ['train_balanced_accuracy',
                'train_tp',
                'train_fp',
                'train_tn',
                'train_fn',
                'train_coverage',
                'test_balanced_accuracy',
                'test_tp',
                'test_fp',
                'test_tn',
                'test_fn',
                'test_coverage',
                'rule_count',
                'run_time']
    row_indexes = []
    if stored_rule_iterations is not None:
        for iter in stored_rule_iterations:
            row_indexes.append('rule_'+str(iter))
    row_indexes.append('rule_post_compact')

    #Gather Results ----------------------------------------------------------------------
    results_list = []
    #Archived Rule Population Evaluations--------------------------
    if stored_rule_iterations is not None:
        for iter in stored_rule_iterations:
            #Training Evaluations-----
            pred_y = exstracs.predict(train_X,rule_pop_iter=iter)
            tn, fp, fn, tp, balanced_accuracy = evaluate_stats(train_y, pred_y)
            cov_y = exstracs.predict_covered(train_X,rule_pop_iter=iter)
            coverage = sum(cov_y)/len(cov_y)
            train_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
            #Testing Evaluations-----
            pred_y = exstracs.predict(test_X,rule_pop_iter=iter)
            tn, fp, fn, tp, balanced_accuracy = evaluate_stats(test_y, pred_y)
            cov_y = exstracs.predict_covered(test_X,rule_pop_iter=iter)
            coverage = sum(cov_y)/len(cov_y)
            test_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
            full_list = train_list + test_list
            #Other Data
            rule_count = len(exstracs.population.pop_set_archive[iter])
            run_time = exstracs.timer.rule_time_archive[iter]
            #Combine into results list
            full_list = full_list + [rule_count,run_time]
            results_list.append(full_list)


    # Final (Compacted) Rule Population Evaluation
    pred_y = exstracs.predict(train_X)
    tn, fp, fn, tp, balanced_accuracy = evaluate_stats(train_y, pred_y)
    cov_y = exstracs.predict_covered(train_X,rule_pop_iter=None)
    coverage = sum(cov_y)/len(cov_y)
    train_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
    #Testing Evaluations-----
    pred_y = exstracs.predict(test_X)
    tn, fp, fn, tp, balanced_accuracy = evaluate_stats(test_y, pred_y)
    cov_y = exstracs.predict_covered(test_X,rule_pop_iter=None)
    coverage = sum(cov_y)/len(cov_y)
    test_list = [balanced_accuracy,tp,fp,tn,fn,coverage]
    full_list = train_list + test_list
    #Other Data
    rule_count = len(exstracs.population.popSet)
    run_time = exstracs.timer.globalTime
    #Combine into results list
    full_list = full_list + [rule_count,run_time]
    results_list.append(full_list)

    #REPORT EVALUATION RESULTS
    results_df = pd.DataFrame(results_list, columns=headers)
    results_df['Row Indexes'] = row_indexes
    results_df.to_csv(outputPath+'/evaluation_summary.csv', index=False)

    #Pickle ExSTraCS object
    #with open(outputPath+'/exstracs.pickle', 'wb') as f:
    #   pickle.dump(exstracs, f)

def evaluate_stats(y_true, y_pred):
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Calculate balanced accuracy
    balanced_accuracy = (sensitivity + specificity) / 2
    return tn, fp, fn, tp, balanced_accuracy

if __name__=="__main__":
    sys.exit(main(sys.argv))