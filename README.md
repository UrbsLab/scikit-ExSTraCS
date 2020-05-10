Master Status: [![Build Status](https://travis-ci.com/UrbsLab/scikit-ExSTraCS.svg?branch=master)](https://travis-ci.com/UrbsLab/scikit-ExSTraCS)

# scikit-ExSTraCS

The scikit-ExSTraCS package includes a sklearn-compatible Python implementation of ExSTraCS 2.0. ExSTraCS 2.0, or Extended Supervised Tracking and Classifying System, implements the core components of a Michigan-Style Learning Classifier System (where the system's genetic algorithm operates on a rule level, evolving a population of rules with each their own parameters) in an easy to understand way, while still being highly functional in solving ML problems. It allows the incorporation of expert knowledge in the form of attribute weights, attribute tracking, rule compaction, and a rule specificity limit, that makes it particularly adept at solving highly complex problems.

In general, Learning Classifier Systems (LCSs) are a classification of Rule Based Machine Learning Algorithms that have been shown to perform well on problems involving high amounts of heterogeneity and epistasis. Well designed LCSs are also highly human interpretable. LCS variants have been shown to adeptly handle supervised and reinforced, classification and regression, online and offline learning problems, as well as missing or unbalanced data. These characteristics of versatility and interpretability give LCSs a wide range of potential applications, notably those in biomedicine. This package is **still under active development** and we encourage you to check back on this repository for updates.

This version of scikit-ExSTraCS is suitable for supervised classification problems only. It has not yet been developed for regression problems. Within these bounds however, scikit-ExSTraCS can be applied to almost any supervised classification data set and supports:

<ul>
  <li>Feature sets that are discrete/categorical, continuous-valued or a mix of both</li>
  <li>Data with missing values</li>
  <li>Binary Classification Problems (Binary Endpoints)</li>
  <li>Multi-class Classification Problems (Multi-class Endpoints)</li>
</ul>

Built into this code, is a strategy to 'automatically' detect from the loaded data, these relevant above characteristics so that they don't need to be parameterized at initialization.

The core Scikit package only supports numeric data. However, an additional StringEnumerator Class is provided that allows quick data conversion from any type of data into pure numeric data, making it possible for natively string/non-numeric data to be run by scikit-XCS.

In addition, powerful data tracking collection methods are built into the scikit package, that continuously tracks features every iteration such as:

<ul>
  <li>Approximate Accuracy</li>
  <li>Average Population Generality</li>
  <li>Macro & Micropopulation Size</li>
  <li>Match Set and Action Set Sizes</li>
  <li>Number of classifiers subsumed/deleted/covered</li>
  <li>Number of crossover/mutation operations performed</li>
  <li>Times for matching, deletion, subsumption, selection, evaluation</li>
</ul>

And many more... These values can then be exported as a csv after training is complete for analysis using the built in "export_iteration_tracking_data" method.

In addition, the package includes functionality that allows the final rule population to be exported as a csv after training.

## Usage
For more information on how to use scikit-ExSTraCS, please refer to the [scikit-ExSTraCS User Guide](https://github.com/UrbsLab/scikit-ExSTraCS/) Jupyter Notebook inside this repository.

## Usage TLDR
```python
#Import Necessary Packages/Modules
from skExSTraCS import ExSTraCS
from skrebate import ReliefF
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

#Load Data Using Pandas
data = pd.read_csv('myDataFile.csv') #REPLACE with your own dataset .csv filename
dataFeatures = data.drop(classLabel,axis=1).values #DEFINE classLabel variable as the Str at the top of your dataset's action column
dataPhenotypes = data[classLabel].values

#Shuffle Data Before CV
formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
np.random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataPhenotypes = formatted[:,-1]

#Get Feature Importance Scores to use as Expert Knowledge (see https://github.com/EpistasisLab/scikit-rebate/ for more details on skrebate package)
relieff = ReliefF()
relieff.fit(dataFeatures,dataPhenotypes)
scores = relieff.feature_importances_

#Initialize ExSTraCS Model
model = ExSTraCS(learning_iterations = 5000,expert_knowledge=scores)

#3-fold CV
print(np.mean(cross_val_score(model,dataFeatures,dataPhenotypes,cv=3)))
```

## License
Please see the repository [license](https://github.com/UrbsLab/scikit-ExSTraCS/blob/master/LICENSE) for the licensing and usage information for scikit-ExSTraCS.

Generally, we have licensed scikit-ExSTraCS to make it as widely usable as possible.

## Installation
scikit-ExSTraCS is built on top of the following Python packages:
<ol>
  <li> numpy </li>
  <li> pandas </li>
  <li> scikit-learn </li>
</ol>

Once the prerequisites are installed, you can install scikit-ExSTraCS with a pip command:
```
pip/pip3 install scikit-ExSTraCS
```
We strongly recommend you use Python 3. scikit-ExSTraCS does not support Python 2, given its depreciation in Jan 1 2020. If something goes wrong during installation, make sure that your pip is up to date and try again.
```
pip/pip3 install --upgrade pip
```

## Contributing to scikit-ExSTraCS
scikit-ExSTraCS is an open source project and we'd love if you could suggest changes!

<ol>
  <li> Fork the project repository to your personal account and clone this copy to your local disk</li>
  <li> Create a branch from master to hold your changes: (e.g. <b>git checkout -b my-contribution-branch</b>) </li>
  <li> Commit changes on your branch. Remember to never work on any other branch but your own! </li>
  <li> When you are done, push your changes to your forked GitHub repository with <b>git push -u origin my-contribution-branch</b> </li>
  <li> Create a pull request to send your changes to the scikit-ExSTraCS maintainers for review. </li>
</ol>

**Before submitting your pull request**

If your contribution changes ExSTraCS in any way, make sure you update the Jupyter Notebook documentation and the README with relevant details. If your contribution involves any code changes, update the project unit tests to test your code changes, and make sure your code is properly commented to explain your rationale behind non-obvious coding practices.

**After submitting your pull request**

After submitting your pull request, Travis CI will run all of the project's unit tests. Check back shortly after submitting to make sure your code passes these checks. If any checks come back failed, do your best to address the errors.

