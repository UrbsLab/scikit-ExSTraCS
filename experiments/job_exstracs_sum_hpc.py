import os 
import sys 
import argparse 
import pickle
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
#sys.path.append('/project/kamoun_shared/code_shared/scikit-heros/')
#from src.skheros.heros import HEROS
#from skheros.heros import HEROS #PIP INSTALL RUN

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--o', dest='outputPath', help='path to target output folder', type=str, default = 'mypath/myOutputFolder') 
    #Dataset Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome label (i.e. class label)', type=str, default = 'Class') 
    parser.add_argument('--il', dest='instanceID_label', help='label of instance ID column (if present)', type=str, default = 'InstanceID')
    parser.add_argument('--el', dest='excluded_column', help='label of another column to drop (if present)', type=str, default = 'Group') 
    #Experiment Parameters
    parser.add_argument('--cv', dest='cv_partitions', help='number of cv partitions', type=int, default = 10) 
    parser.add_argument('--r', dest='random_seeds', help='number of random seeds to run', type=int, default= 30)

    #----------------------------------------------------------------------------------------------
    options=parser.parse_args(argv[1:])
    #Script Parameters
    outputPath = options.outputPath
    #Dataset Parameters
    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column
    #Experiment Parameters
    cv_partitions = options.cv_partitions
    random_seeds = options.random_seeds

    # Get Data Output Folder Name
    folder_name = outputPath.rstrip('/').split('/')[-1]

    # Create CV level summary files
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            for i in range(0,random_seeds):
                seed_level_path = os.path.join(data_level_path,'seed_'+str(i))
                dataframes = []
                for j in range(1,cv_partitions+1):
                    cv_level_path = os.path.join(seed_level_path,'cv_'+str(j))
                    file_path = cv_level_path+'/evaluation_summary.csv'
                    #Load Eval File
                    df = pd.read_csv(file_path)
                    df_X = df.drop('Row Indexes', axis=1)
                    row_names = df['Row Indexes']
                    dataframes.append(df_X)
                #Make averages file
                averaged_df_X = pd.concat(dataframes).groupby(level=0).mean()
                combined_df = pd.concat([row_names, averaged_df_X], axis=1)
                combined_df.to_csv(seed_level_path+'/mean_CV_evaluation_summary.csv', index=False)
                #Make standard deviation file
                sd_df_X = pd.concat(dataframes).groupby(level=0).std()
                combined_df = pd.concat([row_names, sd_df_X], axis=1)
                combined_df.to_csv(seed_level_path+'/sd_CV_evaluation_summary.csv', index=False)

    # Create seed level summary files
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            dataframes = []
            for i in range(0,random_seeds):
                seed_level_path = os.path.join(data_level_path,'seed_'+str(i))
                file_path = seed_level_path+'/mean_CV_evaluation_summary.csv'
                #Load Eval File
                df = pd.read_csv(file_path)
                df_X = df.drop('Row Indexes', axis=1)
                row_names = df['Row Indexes']
                dataframes.append(df_X)
            #Make averages file
            averaged_df_X = pd.concat(dataframes).groupby(level=0).mean()
            combined_df = pd.concat([row_names, averaged_df_X], axis=1)
            combined_df.to_csv(data_level_path+'/mean_seed_evaluation_summary.csv', index=False)
            #Make standard deviation file
            sd_df_X = pd.concat(dataframes).groupby(level=0).std()
            combined_df = pd.concat([row_names, sd_df_X], axis=1)
            combined_df.to_csv(data_level_path+'/sd_seed_evaluation_summary.csv', index=False)

    # Create Global results list for each evaluation point
    header = None
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            eval_point_list = {} #row index dictionary each with list of lists storing results list for every seed and cv combo
            for i in range(0,random_seeds):
                seed_level_path = os.path.join(data_level_path,'seed_'+str(i))
                for j in range(1,cv_partitions+1):
                    cv_level_path = os.path.join(seed_level_path,'cv_'+str(j))
                    file_path = cv_level_path+'/evaluation_summary.csv'
                    #Load Eval File
                    df = pd.read_csv(file_path)
                    header = list(df.columns)
                    header.remove('Row Indexes')
                    for index, row in df.iterrows():
                        #initialize dictionary entry
                        row_index = row['Row Indexes']
                        if row_index not in eval_point_list:
                            eval_point_list[row_index] = []
                        temp_list = list(row)
                        temp_list.remove(row_index)
                        eval_point_list[row_index].append(temp_list)
            #print(eval_point_list)
            #Create global evaluation summary for given dataset
            for each in eval_point_list:
                results = eval_point_list[each]
                df_results = pd.DataFrame(results, columns=header)
                df_results.to_csv(data_level_path+'/all_'+str(each)+'_evaluations.csv', index=False)


    # Create Global (CV average) results list for each evaluation point
    header = None
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            eval_point_list = {} #row index dictionary each with list of lists storing results list for every seed and cv combo
            for i in range(0,random_seeds):
                seed_level_path = os.path.join(data_level_path,'seed_'+str(i))
                file_path = seed_level_path+'/mean_CV_evaluation_summary.csv'
                #Load Eval File
                df = pd.read_csv(file_path)
                header = list(df.columns)
                header.remove('Row Indexes')
                for index, row in df.iterrows():
                    #initialize dictionary entry
                    row_index = row['Row Indexes']
                    if row_index not in eval_point_list:
                        eval_point_list[row_index] = []
                    temp_list = list(row)
                    temp_list.remove(row_index)
                    eval_point_list[row_index].append(temp_list)
            #Create global evaluation summary for given dataset
            for each in eval_point_list:
                results = eval_point_list[each]
                df_results = pd.DataFrame(results, columns=header)
                df_results.to_csv(data_level_path+'/cv_ave_'+str(each)+'_evaluations.csv', index=False)

    #Make testing accuracy boxplots (all runs)
    row_index = ['rule_500','rule_1000','rule_10000','rule_100000','rule_200000','rule_post_compact']
    #for each in eval_point_list:
    #    row_index.append(each)
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            accuracy_values = []
            for each in row_index: #each eval iteration
                #Load Eval File
                file_path = data_level_path+'/all_'+str(each)+'_evaluations.csv'
                df = pd.read_csv(file_path)
                accuracy_values.append(df['test_balanced_accuracy'])
            # Create a boxplot for all 10 accuracy columns
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=accuracy_values)
            # Set the labels for the x-axis to represent each file
            plt.xticks(ticks=range(len(row_index)), labels=row_index,rotation=90)
            # Title and labels
            plt.xlabel("Evaluation Points")
            plt.ylabel("Balanced Testing Accuracy")
            plt.savefig(data_level_path+'/boxplot_testing_accuracy_all.png', bbox_inches="tight")


    #Make  rule count boxplots (all runs)
    for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(outputPath,entry)):
            data_level_path = os.path.join(outputPath,entry)
            count_values = []
            for each in row_index: #each eval iteration
                #Load Eval File
                file_path = data_level_path+'/all_'+str(each)+'_evaluations.csv'
                df = pd.read_csv(file_path)
                count_values.append(df['rule_count'])
            # Create a boxplot for all 10 accuracy columns
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=count_values)
            # Set the labels for the x-axis to represent each file
            plt.xticks(ticks=range(len(row_index)), labels=row_index,rotation=90)
            # Title and labels
            plt.xlabel("Evaluation Points")
            plt.ylabel("Rule Count")
            plt.savefig(data_level_path+'/boxplot_rule_count_all.png', bbox_inches="tight")

    # Identify how many ideal MUX rules were identified at end of phase 1 population (pre and post compaction)
    if 'multiplexer' in outputPath: #only do if we are evaluating MUX problem
        mux_output_folders = {'A_multiplexer_6_bit_500_inst':6,'B_multiplexer_11_bit_5000_inst':11,'C_multiplexer_20_bit_10000_inst':20,'D_multiplexer_37_bit_10000_inst':37,'E_multiplexer_70_bit_20000_inst':70}
        header = ['Dataset','Seed','CV','Ideal Count','Ideal Proportion']
        resultsList = []
        for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
            if os.path.isdir(os.path.join(outputPath,entry)):
                data_level_path = os.path.join(outputPath,entry)
                for i in range(0,random_seeds):
                    seed_level_path = os.path.join(data_level_path,'seed_'+str(i))
                    dataframes = []
                    for j in range(1,cv_partitions+1):
                        cv_level_path = os.path.join(seed_level_path,'cv_'+str(j))
                        file_path = cv_level_path+'/rule_pop.csv' #load rule population file
                        #Load Eval File
                        df = pd.read_csv(file_path)
                        # Get ideal rule set list
                        ideal_rules = gen_ideal_rules(mux_output_folders[entry])
                        for each in ideal_rules:
                            exstracs_mux_translator(each,mux_output_folders[entry])
                        ideal_rule_count = len(ideal_rules)
                        found_ideal_rules = 0
                        #Check Rule Pop for number and proportion of ideal rules
                        for index, row in df.iterrows():
                            c_index = row['Specified Attribute Names']
                            c_values = row['Specified Values']
                            outcome = row['Class']
                            combined_rule = [c_index,c_values,int(outcome)]
                            if combined_rule in ideal_rules:
                                found_ideal_rules += 1
                        found_ideal_proportion = found_ideal_rules / float(ideal_rule_count)
                        tempList = [entry, i, j, found_ideal_rules,found_ideal_proportion]
                        resultsList.append(tempList)
        df_results = pd.DataFrame(resultsList,columns=header)
        df_results.to_csv(outputPath+'/phase1_mux_ideal_success.csv', index=False) 
        #Make quick average summary file
        header = ['Dataset','Ideal Count','Ideal Proportion']
        resultsList = []
        for entry in os.listdir(outputPath): #for each subfolder within target dataset folder
            if entry != 'phase1_mux_ideal_success.csv' and entry != 'phase1_mux_ideal_success_average.csv':
                average_count = df_results.loc[df_results['Dataset'] == entry, 'Ideal Count'].mean()
                average_proportion = df_results.loc[df_results['Dataset'] == entry, 'Ideal Proportion'].mean()
                resultsList.append([entry,average_count,average_proportion])
        df_results = pd.DataFrame(resultsList,columns=header)
        df_results.to_csv(outputPath+'/phase1_mux_ideal_success_average.csv', index=False) 

def gen_ideal_rules(mux):
    address_bits = {6:2, 11:3, 20:4, 37:5, 70:6, 135:7}
    ideal_list = []
    register_bits = mux - address_bits[mux]
    for i in range(0,register_bits): #each unique index list
        #ZERO OUTCOME
        index_list = []
        value_list = []
        #Build index list
        for j in range(0,address_bits[mux]):
            index_list.append(j)
        index_list.append(i+address_bits[mux])
        #Build value list
        value_list = (int_to_binary_list(i,address_bits[mux]))
        value_list.append(0)
        ideal_list.append([str(index_list),str(value_list),0])
        #ONE OUTCOME
        value_list = []
        #Build value list
        value_list = (int_to_binary_list(i,address_bits[mux]))
        value_list.append(1)
        ideal_list.append([str(index_list),str(value_list),1])

    return ideal_list

def get_mux_index_dictionary(mux):
    address_bits = {6:2, 11:3, 20:4, 37:5, 70:6, 135:7}
    register_bits = mux - address_bits[mux]
    mux_dict = {}
    register_counter = 0
    for i in range(0,mux):
        if i < address_bits[mux]:
            mux_dict[i] = 'A_'+str(i)
        else:
            mux_dict[i] = 'R_'+str(register_counter)
            register_counter += 1
    return mux_dict

def exstracs_mux_translator(ideal_rule, mux):
    #Deal with feature indexes
    mux_dict = get_mux_index_dictionary(mux)
    temp_index_list = ast.literal_eval(ideal_rule[0])
    index_string = ''#'"'
    for i in range(0,len(temp_index_list)):
    #for each in temp_index_list:
        if i < len(temp_index_list) - 1:
            index_string += mux_dict[temp_index_list[i]]
            index_string += ', '
        else:
            index_string += mux_dict[temp_index_list[i]]
    #index_string += '"'
    ideal_rule[0] = index_string

    #Deal with specified values
    ideal_rule[1] = ", ".join(map(str, ast.literal_eval(ideal_rule[1])))
    #ideal_rule[1] = '"'+str(ideal_rule[1])+'"'

def int_to_binary_list(num, n):
    """Convert an integer to an n-digit binary list of 0s and 1s."""
    return [int(bit) for bit in format(num, f'0{n}b')]

#Heros - for each dataset make a file for each 
    #for each row index (keep name for file)
        #save the cv average for each seed (as new row)
        #save as file
        #save the raw scores for each seed and cv
        #save as file

#for each dataset
    #a file for accuracy and one for coverage
        #each column is a time point (up to final model (test selected)) - for boxplots

#Make boxplot of each part of heros for model times up to test-selected model final (of test accuracies and separate for rule count)

#For each dataset
    #HEROS VS EXSTRACS RULE POP: comparing - rule populations (heros, exstracs) post compaction (2 groups) - do this for nu =1 and 10
        #training, testing, and rule count, runtime
        #stats compare each

    #HEROS (VALUE OF PHASE 2):comparing - heros phase 1 (pre and post compation) to phase 2 (default model final, and test selected model final) (4 groups)
        #training, testing, rule count, coverage, runtime
        #stats compare each

    #HEROS (VALUE OF MODEL INIT): comparing -heros phase 2 (random vs. target_acc init) - all default and test selected time points
        #training, testing, rule count, coverage, runtime
        #stats compare each

    #MULTIPLEXER (nu= 1 or 10): 


#apply statistical comparisons

if __name__=="__main__":
    sys.exit(main(sys.argv))