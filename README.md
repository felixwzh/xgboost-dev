
# MT-GBDT	
This is the implementation of paper  Multi-Task Gradient Boosting Decision Tree on [XGBoost](https://github.com/dmlc/xgboost).	
install the python package of xgboost.	

```	
make	
cd python-package	
sudo python setup.py install	
```	

About how to use MT-GBDT in python, see details [here](demo/multi_task_binary_classification/)

If you find this repository helpful in your research, kindly consider citing the paper mentioned below:
```
@inproceedings{chen2021task,
  title={Task-wise split gradient boosting trees for multi-center diabetes prediction},
  author={Chen, Mingcheng and Wang, Zhenghui and Zhao, Zhiyun and Zhang, Weinan and Guo, Xiawei and Shen, Jian and Qu, Yanru and Lu, Jieli and Xu, Min and Xu, Yu and others},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={2663--2673},
  year={2021}
```
