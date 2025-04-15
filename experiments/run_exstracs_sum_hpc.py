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

" HEROS Run "

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--w', dest='writepath', help='path to folder where all outputs from jobs/scripts will be saved', type=str, default = 'mypath/myWritePath') 
    parser.add_argument('--o', dest='outputfolder', help='unique folder name for this analysis', type=str, default = 'myAnalysis') 
    #Dataset Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome label (i.e. class label)', type=str, default = 'Class') 
    parser.add_argument('--il', dest='instanceID_label', help='label of instance ID column (if present)', type=str, default = 'InstanceID')
    parser.add_argument('--el', dest='excluded_column', help='label of another column to drop (if present)', type=str, default = 'Group') 
    #Experiment Parameters
    parser.add_argument('--cv', dest='cv_partitions', help='number of cv partitions', type=int, default = 10) 
    parser.add_argument('--r', dest='random_seeds', help='number of random seeds to run', type=int, default= 30)
    #HPC parameters
    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str, default='LSF')
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 4)
    parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal')
    #----------------------------------------------------------------------------------------------
    options=parser.parse_args(argv[1:])
    #Script Parameters
    writepath = options.writepath
    outputfolder = options.outputfolder
    #Dataset Parameters
    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column
    #Experiment Parameters
    cv_partitions = options.cv_partitions
    random_seeds = options.random_seeds
    #HPC parameters
    run_cluster = options.run_cluster
    reserved_memory = options.reserved_memory
    queue = options.queue

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
    if run_cluster == 'LSF':
        submit_lsf_cluster_job(scratchPath,logPath,reserved_memory,queue,base_output_path_0,outcome_label,instanceID_label,excluded_column,cv_partitions,random_seeds,outputfolder)
    elif run_cluster == 'SLURM':
        submit_slurm_cluster_job(scratchPath,logPath,reserved_memory,queue,base_output_path_0,outcome_label,instanceID_label,excluded_column,cv_partitions,random_seeds,outputfolder)
    else:
        print('ERROR: Cluster type not found')
    print(str(1)+' jobs submitted successfully')

#UPenn Cluster
def submit_lsf_cluster_job(scratchPath,logPath,reserved_memory,queue,base_output_path_0,outcome_label,instanceID_label,excluded_column,cv_partitions,random_seeds,outputfolder): 
    job_ref = str(time.time())
    job_name = 'Exstracs_summary_'+outputfolder+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('python job_exstracs_sum_hpc.py'+' --o '+str(base_output_path_0)+' --ol '+str(outcome_label) +' --il '+str(instanceID_label) +' --el '+str(excluded_column)+' --cv '+str(cv_partitions)+' --r '+str(random_seeds)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_path)

#Cedars Cluster
def submit_slurm_cluster_job(scratchPath,logPath,reserved_memory,queue,base_output_path_0,outcome_label,instanceID_label,excluded_column,cv_partitions,random_seeds,outputfolder): 
    job_ref = str(time.time())
    job_name = 'Exstracs_summary_'+outputfolder+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python job_exstracs_sum_hpc.py'+' --o '+str(base_output_path_0)+' --ol '+str(outcome_label) +' --il '+str(instanceID_label) +' --el '+str(excluded_column)+' --cv '+str(cv_partitions)+' --r '+str(random_seeds)+'\n')
    sh_file.close()
    os.system('sbatch ' + job_path)

if __name__=="__main__":
    sys.exit(main(sys.argv))