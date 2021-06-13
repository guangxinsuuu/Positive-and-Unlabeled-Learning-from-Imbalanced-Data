# ImbalancedPU
Code for the paper named "Positive-Unlabeled Learning from Imbalanced Data"[1] which has been accepted by IJCAI-21

There are two folders named "ImbalancedSelfPU" and "ImbalancednnPU": 

## ImbalancedSelfPU

This framework combines our proposed method named ImbalancedPU and the framework depends on the Self-PU [2] which uses the additional labeled data, in a meta-learning fashion and other “self”-oriented building blocks.

* ```util.py``` The related ImbalancedPU loss functions are used here which are changed based on the Self-PU  [2] framework.

* ```train_with_meta.py``` This is an example code of nnPU learning and uPU learning. 
Dataset is CIFAR10 [3] preprocessed in such a way that artifacts form the P class and living things form the N class.
The default setting is 1000 P data and 50000 U data of CIFAR10, and the class prior is the ratio of P class data in U data.
