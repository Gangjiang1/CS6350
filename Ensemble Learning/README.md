To run the program, either click the run.sh file or open a command prompt and change the directory to the current folder. 
Then type  

`python3 Driver.py <directory of training set> <directory of test set> <algorithm> <max iteration>`  

`<algorithm>` allows you to 
choose the algorithm to build the model; you can choose either `adaboost`, `bagging`, or 
`randomforest`. `<max iteration>` allows you to set the maxmimum iterations to run the model.  

Optional arguments can be added to the command:  

`python3 driver <directory of training set> <directory of test set> <algorithm> <max iteration> [<max samples> <max features>]`  

`<max samples>` is the number of samples to draw at each iteration for both bagging and random forest (default to 500).
`<max features>` is the number of features to draw at each iteration for random forest algorithm (default to 3).  

