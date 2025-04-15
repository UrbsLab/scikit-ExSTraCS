import sys
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as scs
import random
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import csv
import time
import argparse

" Exstracs Run "

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datafolder', help='path to data folder includind data subfolders ', type=str, default = 'mypath/myDataFolder') 
    parser.add_argument('--w', dest='writepath', help='path to folder where all outputs from jobs/scripts will be saved', type=str, default = 'mypath/myWritePath') 
    parser.add_argument('--o', dest='outputfolder', help='unique folder name for this analysis', type=str, default = 'myAnalysis') 
    parser.add_argument('--ekf', dest='ekfolder', help='path to the folder containing ek scores for datasets', type=str, default = 'mypath/ekScores') 
    #Dataset Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome label (i.e. class label)', type=str, default = 'Class') 
    parser.add_argument('--il', dest='instanceID_label', help='label of instance ID column (if present)', type=str, default = 'InstanceID')
    parser.add_argument('--el', dest='excluded_column', help='label of another column to drop (if present)', type=str, default = 'Group') 
    #Experiment Parameters
    parser.add_argument('--cv', dest='cv_partitions', help='number of cv partitions', type=int, default = 10) 
    parser.add_argument('--r', dest='random_seeds', help='number of random seeds to run', type=int, default= 30)
    #Critical Exstracs Parameters
    parser.add_argument('--it', dest='learning_iterations', help='number of rule training cycles', type=int, default=100000)
    parser.add_argument('--ps', dest='N', help='maximum micro rule population size', type=int, default=1000)
    parser.add_argument('--nu', dest='nu', help='power parameter', type=int, default=1)
    #Other Exstracs Parameters
    #parser.add_argument('--ft', dest='do_attribute_tracking', help='feature tracking mechanism', type=bool, default=True)
    parser.add_argument('--ft', dest='do_attribute_tracking', help='feature tracking mechanism', action='store_true')
    #parser.add_argument('--ff', dest='do_attribute_feedback', help='boolean flag to use feature tracking feedback', type=bool, default=False)
    parser.add_argument('--ff', dest='do_attribute_feedback', help='boolean flag to use feature tracking feedback', action='store_true')
    parser.add_argument('--c', dest='compaction', help='rule-compaction strategy', type=str, default='QRF')
    #parser.add_argument('--ta', dest='track_accuracy_while_fit', help='boolean flag to use accuracy tracking', type=bool, default=True)
    parser.add_argument('--ta', dest='track_accuracy_while_fit', help='boolean flag to use accuracy tracking', action='store_true')
    parser.add_argument('--sr', dest='stored_rule_iterations', help='comma-separated string indicating rule pop iterations to run full evaluation', type=str, default=None)
    parser.add_argument('--rs', dest='random_state', help='random state seed', type=int, default=42)
    #HPC parameters
    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str, default='LSF')
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 4)
    parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal')
    #parser.add_argument('--check', dest='check', help='boolean flag to check and report on what jobs have not yet completed', type=bool, default= False)
    #parser.add_argument('--resub', dest='resubmit', help='boolean flag to resubmit incomplete jobs', type=bool, default= False)
    parser.add_argument('--check', dest='check', help='boolean flag to check and report on what jobs have not yet completed', action='store_true')
    parser.add_argument('--resub', dest='resubmit', help='boolean flag to resubmit incomplete jobs', action='store_true')
    #----------------------------------------------------------------------------------------------
    options=parser.parse_args(argv[1:])
    #Script Parameters
    datafolder = options.datafolder #full path to target dataset
    writepath = options.writepath
    outputfolder = options.outputfolder
    ekfolder = options.ekfolder
    #Dataset Parameters
    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column
    #Experiment Parameters
    cv_partitions = options.cv_partitions
    random_seeds = options.random_seeds
    #Critical Exstracs Parameters
    learning_iterations = options.learning_iterations
    N = options.N
    nu = options.nu
    #Other Exstracs Parameters
    do_attribute_tracking = options.do_attribute_tracking
    do_attribute_feedback = options.do_attribute_feedback
    compaction = options.compaction
    track_accuracy_while_fit = options.track_accuracy_while_fit
    if stored_rule_iterations is None or stored_rule_iterations == 'None':
        stored_rule_iterations = None
    else:
        stored_rule_iterations = options.stored_rule_iterations
    random_state = options.random_state
    #HPC parameters
    run_cluster = options.run_cluster
    reserved_memory = options.reserved_memory
    queue = options.queue
    check = options.check
    resubmit = options.resubmit

    algorithm = 'ExSTraCS'

    #Folder Management------------------------------
    #Main Write Path-----------------
    if not os.path.exists(writepath):
        os.mkdir(writepath)  
    #Output Path--------------------
    base_output_path_0 = writepath+'/output/'
    if not os.path.exists(base_output_path_0):
        os.mkdir(base_output_path_0) 
    #Scratch Path-------------------- 
    scratchPath = writepath+'/scratch'
    if not os.path.exists(scratchPath):
        os.mkdir(scratchPath) 
    #LogFile Path--------------------
    logPath = writepath+'/logs'
    if not os.path.exists(logPath):
        os.mkdir(logPath) 
    base_output_path_0 = base_output_path_0+algorithm+'_'+outputfolder
    if not os.path.exists(base_output_path_0):
        os.mkdir(base_output_path_0) 

    # Experiment loop to submit jobs (datasets, random seeds, cv partitions)
    jobCount = 0
    missing_count = 0
    for entry in os.listdir(datafolder): #for each subfolder within target dataset folder
        if os.path.isdir(os.path.join(datafolder,entry)):
            datapath = os.path.join(datafolder,entry)
            #Specify output folder path
            base_output_path_1 = base_output_path_0+'/'+entry
            if not os.path.exists(base_output_path_1):
                os.mkdir(base_output_path_1) 

        for i in range(0,random_seeds):
            target_random_seed = None
            if random_seeds > 1:
                target_random_seed = i
            else: 
                target_random_seed = random_state
            #Specify output folder path
            base_output_path_2 = base_output_path_1+'/'+'seed_'+str(i)
            if not os.path.exists(base_output_path_2):
                os.mkdir(base_output_path_2) 

            for j in range(1,cv_partitions+1):
                #Specify dataset path
                full_data_path = datapath+'/'+str(entry)+'_CV_Train_'+str(j)+'.txt'
                full_data_name = entry+'_CV_Train_'+str(j)
                #Specify output folder path
                outputPath = base_output_path_2+'/'+'cv_'+str(j)
                if not os.path.exists(outputPath):
                    os.mkdir(outputPath) 

                if not check: #Regular Job submission run
                    if run_cluster == 'LSF':
                        submit_lsf_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,ekfolder,outcome_label,instanceID_label,excluded_column,learning_iterations,N,nu,do_attribute_tracking,do_attribute_feedback,compaction,track_accuracy_while_fit,stored_rule_iterations,target_random_seed)
                        jobCount +=1
                    elif run_cluster == 'SLURM':
                        submit_slurm_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,ekfolder,outcome_label,instanceID_label,excluded_column,learning_iterations,N,nu,do_attribute_tracking,do_attribute_feedback,compaction,track_accuracy_while_fit,stored_rule_iterations,target_random_seed)
                        jobCount +=1
                    else:
                        print('ERROR: Cluster type not found')
                else: #check what runs have completed (based on last file generated by jobs)
                    target_file_path = outputPath+'/evaluation_summary.csv'
                    if not os.path.exists(target_file_path):
                        print('Missing: '+str(outputPath))
                        missing_count += 1
                        if resubmit:
                            if run_cluster == 'LSF':
                                submit_lsf_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,ekfolder,outcome_label,instanceID_label,excluded_column,learning_iterations,N,nu,do_attribute_tracking,do_attribute_feedback,compaction,track_accuracy_while_fit,stored_rule_iterations,target_random_seed)
                                jobCount +=1
                            elif run_cluster == 'SLURM':
                                submit_slurm_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,ekfolder,outcome_label,instanceID_label,excluded_column,learning_iterations,N,nu,do_attribute_tracking,do_attribute_feedback,compaction,track_accuracy_while_fit,stored_rule_iterations,target_random_seed)
                                jobCount +=1
                            else:
                                print('ERROR: Cluster type not found')

    print(str(jobCount)+' jobs submitted successfully')
    if check:
        print(str(missing_count)+' jobs incomplete')

#UPenn Cluster
def submit_lsf_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,ekfolder,outcome_label,instanceID_label,excluded_column,learning_iterations,N,nu,do_attribute_tracking,do_attribute_feedback,compaction,track_accuracy_while_fit,stored_rule_iterations,target_random_seed): 
    job_ref = str(time.time())
    job_name = 'Exstracs_'+full_data_name+'_seed_'+str(target_random_seed)+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('python job_exstracs_hpc.py'+' --d '+str(full_data_path)+' --o '+str(outputPath)+' --ekf '+str(ekfolder)+' --ol '+str(outcome_label) +' --il '+str(instanceID_label) +' --el '+str(excluded_column)+' --it '+str(learning_iterations)+' --ps '+str(N)+' --nu '+str(nu)+' --ft '+str(do_attribute_tracking)+' --ff '+str(do_attribute_feedback)+' --c '+str(compaction)+' --ta '+str(track_accuracy_while_fit)+' --sr '+str(stored_rule_iterations)+' --rs '+str(target_random_seed)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_path)

#Cedars Cluster
def submit_slurm_cluster_job(scratchPath,logPath,reserved_memory,queue,full_data_name,outputPath,full_data_path,ekfolder,outcome_label,instanceID_label,excluded_column,learning_iterations,N,nu,do_attribute_tracking,do_attribute_feedback,compaction,track_accuracy_while_fit,stored_rule_iterations,target_random_seed): 
    job_ref = str(time.time())
    job_name = 'Exstracs_'+full_data_name+'_seed_'+str(target_random_seed)+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python job_exstracs_hpc.py'+' --d '+str(full_data_path)+' --o '+str(outputPath)+' --ekf '+str(ekfolder)+' --ol '+str(outcome_label) +' --il '+str(instanceID_label) +' --el '+str(excluded_column)+' --it '+str(learning_iterations)+' --ps '+str(N)+' --nu '+str(nu)+' --ft '+str(do_attribute_tracking)+' --ff '+str(do_attribute_feedback)+' --c '+str(compaction)+' --ta '+str(track_accuracy_while_fit)+' --sr '+str(stored_rule_iterations)+' --rs '+str(target_random_seed)+'\n')
    sh_file.close()
    os.system('sbatch ' + job_path)

if __name__=="__main__":
    sys.exit(main(sys.argv))