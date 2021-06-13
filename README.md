# ImbalancedPU
Code for the paper named "Positive-Unlabeled Learning from Imbalanced Data"[1] which has been accepted by IJCAI-21

There are two folders named "ImbalancedSelfPU" and "ImbalancednnPU": 

* ```pu_loss.py``` has a chainer implementation of the risk estimator for non-negative PU (nnPU) learning and unbiased PU (uPU) learning. 
* ```train.py``` is an example code of nnPU learning and uPU learning. 
Dataset are MNIST [3] preprocessed in such a way that even digits form the P class and odd digits form the N class and
CIFAR10 [4] preprocessed in such a way that artifacts form the P class and living things form the N class.
The default setting is 100 P data and 59900 U data of MNIST, and the class prior is the ratio of P class data in U data.
