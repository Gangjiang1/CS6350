To run the program, either click the run.sh file or open a command prompt and change the directory to the current folder. 
Then type  

`python3 LMS.py <directory of training set> <directory of test set> <algorithm> [<learning rate> <tolerance> <max iteration>]`  

`<algorithm>` allows you to choose the algorithm to build the model; you can choose either `gradient_descent` or 
`stochastic_gradient_descent`. `<learning rate>` allows you to set the learning rate for the iterations (default to 1). `<tolerance>` allows
you to set minimum difference of cost between two iterations for the model to keep running (default to 1e-6). `<max iteration>` allows 
you to set the maxmimum iterations to run the model (default to 20000).