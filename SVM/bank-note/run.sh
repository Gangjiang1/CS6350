# primal svm with schedule gammat = gamma0 / (1 + gamma0/a) * t)
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv svm 0.1 100/873 100 linear 1 0.5
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv svm 0.1 500/873 100 linear 1 0.5
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv svm 0.1 700/873 100 linear 1 0.5

# primal svm with schedule gammat = gamma0 / (1 + t)
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv svm 0.1 100/873 100 linear 2
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv svm 0.1 500/873 100 linear 2
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv svm 0.1 700/873 100 linear 2

# dual svm with linear kernel
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 0.1 100/873 100
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 0.1 500/873 100
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 0.1 700/873 100

# dual svm with gaussian kernel
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 0.1 100/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 0.5 100/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 1 100/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 5 100/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 100 100/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 0.1 500/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 0.5 500/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 1 500/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 5 500/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 100 500/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 0.1 700/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 0.5 700/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 1 700/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 5 700/873 100 gaussian
python3 Driver.py ./bank-note/train.csv ./bank-note/test.csv kernel_svm 100 700/873 100 gaussian
