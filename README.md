# BBL-WPP
This code solves short term wind power prediction problem using multiple sampling resolution data. For more details, please see our paper [A Bilateral Branch Learning Paradigm for Short Term Wind Power Prediction with Data of Multiple Sampling Resolutions](https://ieeexplore.ieee.org/abstract/document/9537638) which has been accepted at JCLP. If this code is useful for your work, please cite our paper:

```
@article{liu2022bilateral,
  title={A bilateral branch learning paradigm for short term wind power prediction with data of multiple sampling resolutions},
  author={Liu, Hong and Zhang, Zijun},
  journal={Journal of Cleaner Production},
  pages={134977},
  year={2022},
  publisher={Elsevier}
}
```

## Dependencies

* python = 3.6.3
* NumPy
* Scipy
* PyTorch = 1.7
* xgboost
* scikit-learn
* optuna
* pyearth

## Data

Since the github cannot process the data larger than 4GB, please download the files 'data_wf1_7s' from this [google drive link](https://drive.google.com/file/d/1uh96xI-KL4ANGpqi0QqPLscFYzLZibHY/view?usp=sharing) and 'data_wf1_10min' from this [google drive link](https://drive.google.com/file/d/1TVr2rd4eGP3DP8FeWNWYoNpcifrpZWDn/view?usp=sharing), and place them to the folder 'wf1'.



## Quick Start

For using LASSO as the feature selection and prediction model, and single sampling resolution data:

```
python code/run.py --fs lasso --fp lasso
```

For using random forest as the feature selection and decision tree as prediction model, and multiple sampling resolution data:

```
python code/run.py --fs rf --fp dt --need_fh
```

