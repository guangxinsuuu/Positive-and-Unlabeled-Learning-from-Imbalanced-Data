from without_meta import train_nnpu_sigmoid
import sys

#num_initial_pos, learning_rate, weight_decay, num_batches, seed

#xx.py = sys.argv[0]
#'num_initial_pos' = sys.argv[1]
num_initial_pos = int(sys.argv[2])
#'learning_rate' = sys.argv[3]
learning_rate = int(sys.argv[4])
#'weight_decay' = sys.argv[5]
weight_decay = int(sys.argv[6])
#'num_batches' = sys.argv[7]
num_batches = int(sys.argv[8])
#'seed' = sys.argv[9]
seed = int(sys.argv[10])
#'label_num' = sys.argv[11]
label_num = int(sys.argv[12])

#python train_nnpu_sigmoid_main.py -num_initial_pos 1000 -learning_rate 3 -weight_decay 8 -num_batches 3 -seed 1
train_nnpu_sigmoid(num_initial_pos, learning_rate, weight_decay, num_batches, seed, label_num)