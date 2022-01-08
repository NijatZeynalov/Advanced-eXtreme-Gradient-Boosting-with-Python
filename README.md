# Advanced eXtreme Gradient Boosting with Python

Xgboost uses a combination of parallelization, tree pruning, hardware optimization,regularization, sparsity awareness,weighted quartile sketch and cross validation. Original idea of boosting came from Michael Kearns (Thoughts on Hypothesis boosting), he suggested if a weak learner can be modified to enhanced predictions in boosting.

A weak learner was defined as a model whose performance is just better than random chance. Hypothesis boosting idea is simple yet powerful, it suggests filter observations that a weak learner can handle and focus on developing new weak learners who can handle remaining tough observations. Overall you get a highly accurate model.

In this project I have implemented XGBoost algorithm on  "Otto Group Product Classification Challenge" competition dataset. There are a total of 93 numerical features, which represent counts of different events. Each target category represents one of our most important product categories (like fashion, electronics, etc.). I just have used train.csv dataset and split its own to train and test datasets.

XGBoost provides a wrapper class to allow models to be treated like classifiers or regressors in the scikit-learn framework. So we can easily use it. I also have used specific features of XGBoost, such as early stopping, monitor training. The XGBoost model can evaluate and report on the performance on a test set for the model
during training. 

We can retrieve the performance of the model on the evaluation dataset and plot it to get
insight into how learning unfolded while training. We provide an array of X and y pairs to the
eval metric argument when fitting our XGBoost model. In addition to a test set, we can also
provide the training dataset. This will provide a report on how well the model is performing on
both training and test sets during training. 

Two plots are created. The first shows the error of the XGBoost model for
each epoch on the training and test datasets.


![alt text](https://github.com/NijatZeynalov/Advanced-eXtreme-Gradient-Boosting-with-Python/blob/main/img/model_error.jpeg?raw=true)


The XGBoost library for gradient boosting uses is designed for efficient multi-core parallel processing which allows it to efficiently use all of the CPU cores in your system when training. We can confirm that XGBoost multi-threading support is working by building a
number of different XGBoost models, specifying the number of threads and timing how long
it takes to build each model. The trend will both show we that multi-threading support is
enabled and give you an indication of the effect it has when building model.

![alt text](https://github.com/NijatZeynalov/Advanced-eXtreme-Gradient-Boosting-with-Python/blob/main/img/model_performance.png?raw=true)



We can see a nice trend in the decrease in execution time as the number of threads is
increased. 

