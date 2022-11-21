To run the model, you may simply run the `run.sh` file, or open a command prompt and type  

`python3 Driver.py <directory:training set> <directory:test set> <algorithm> <learning rate/gamma> <C-value> <maximum iteration>
[<kernel function> <learning rate schedule> <a-value>]`

For `<algorithm>`, type either `svm` or `kernel_svm`.
Any other input will raise an error.  

The last 3 arguments are optional. `<kernel function>` is default to linear kernel. For Gaussian kernel, type `gaussian`.
Under Gaussian kernel, the value of the variance term gamma is taken from the argument `<learning rate/gamma>`.
`<learning rate schedule>` is default to 1, which follows `lrt_t = lrt0 / (1 + (lrt0/a) * t)`, and the `<a-value>` is 
default to 0.5. To run the model with the schedule `lrt_t = lrt0 / (1 + t)`, set `<learning rate schedule>` to 2.

Note: if you are running primal SVM and want to set learning rate schedule, please manually specify `<kernel function>` 
to `linear`

Primal SVM and linear kernelized SVM will return weights and prediction errors. Gaussian kernelized SVM will return
prediction errors only.

A `SettingWithCopyWarning` may appear when loading the datasets. However, it does not affect the performance of the
program.
