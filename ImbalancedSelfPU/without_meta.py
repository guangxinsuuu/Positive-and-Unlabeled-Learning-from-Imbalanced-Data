import torch
import numpy as np
from torch import optim
import scipy.io as scio
from torch.utils.data import sampler
from label import load_dataset
from model import CNN
from unbal_loss import OversampledPULossFunc, OversampledNNPULossFunc, NNZeroOneTrain, ZeroOneTest
from unbal_loss import Auc_loss, PUF1, PUPrecision, PURecall
from set_random_seed import set_seed

def train_nnpu_sigmoid(num_initial_pos, learning_rate, weight_decay, num_batches, seed,label_num):
    epochs = 200
    bn_parameter = 0.9
    prior_prime = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'

    set_seed(seed)

    # Define the classifier/model
    model = CNN()
    model.to(device)

    # Define the criterion
    # For uPU
    criterion_upu = OversampledPULossFunc()
    # For nnPU
    criterion_nn = OversampledNNPULossFunc()
    # For 01 training
    criterion_training_01 = NNZeroOneTrain()
    # For 01 testing
    criterion_test_01 = ZeroOneTest()
    # For precision
    criterion_precision = PUPrecision()
    # For recall
    criterion_recall = PURecall()
    # For F1
    criterion_F1 = PUF1()
    # For Auc
    criterion_Auc = Auc_loss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1*10**(-learning_rate), weight_decay=5*10**(-weight_decay))
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800], gamma=0.1)

    # Load data set
    xy_pos_train, xy_unlabel_train, xy_pos_test, xy_neg_test, prior = load_dataset("cifar10", 1000, 50000, seed, label_num)


    trainloader_positive = torch.utils.data.DataLoader(xy_pos_train,

                                                       batch_size=10,
                                                       shuffle=True)
    trainloader_unlabeled = torch.utils.data.DataLoader(xy_unlabel_train,
                                                        batch_size=500,
                                                        shuffle=True)
    testloader_positive = torch.utils.data.DataLoader(xy_pos_test,

                                                      batch_size=int(len(xy_pos_test)),
                                                      shuffle=False)
    testloader_negative = torch.utils.data.DataLoader(xy_neg_test,
                                                      batch_size=int(len(xy_neg_test)),
                                                      shuffle=False)


    test_01_losses_oversample = []
    training_01_losses_oversample = []
    test_precions = []
    test_recalls = []
    test_F1s = []
    training_01_losses = []
    test_01_losses = []
    running_losses = []
    training_sigmoid_losses = []
    Auc_losses = [ ]

    for epoch in range(epochs):
        # scheduler.step()
        running_loss = []
        for (train_positive, train_unlabeled) in zip(trainloader_positive, trainloader_unlabeled):

            # Get positive data and unlabeled data
            positive_instances, positive_labels = train_positive
            unlabeled_instances, unlabeled_labels = train_unlabeled

            # Put them together in case of batch norm
            train_instances = torch.cat((positive_instances, unlabeled_instances), 0)
            # flatten instances

            # calculate the output and get train and unlabeled separated
            train_instances = train_instances.to(device)
            train_outputs = model.forward(train_instances)
            train_outputs_positive = train_outputs[:positive_instances.shape[0]]
            train_outputs_unlabeled = train_outputs[positive_instances.shape[0]:]

            # Calculate objective and train
            loss_nn = criterion_nn(train_outputs_positive, train_outputs_unlabeled, prior, prior_prime)
            loss_upu = criterion_upu(train_outputs_positive, train_outputs_unlabeled, prior, prior_prime)
            optimizer.zero_grad()
            if loss_nn > 0:
                loss = loss_upu
            else:
                loss = - loss_nn
            loss.backward()
            optimizer.step()

            one_iteration_running_loss = max(loss_nn.item(), 0) + loss_upu.item() - loss_nn.item()
            running_loss.append(one_iteration_running_loss)

            # Calculate various losses
            # objective
        running_losses.append(np.mean(running_loss))

        with torch.no_grad():
            model.eval()
            model.to(device)

            training_01_loss = []
            training_01_loss_oversample = []
            for (train_positive, train_unlabeled) in zip(trainloader_positive, trainloader_unlabeled):
                # Get positive data and unlabeled data
                positive_instances, positive_labels = train_positive
                unlabeled_instances, unlabeled_labels = train_unlabeled

                positive_instances = positive_instances.to(device)
                unlabeled_instances = unlabeled_instances.to(device)

                train_outputs_positive = model(positive_instances)
                train_outputs_unlabeled = model(unlabeled_instances)

                # traing 01 loss
                training_01_loss.append(
                    criterion_training_01(train_outputs_positive, train_outputs_unlabeled, prior).item())
                training_01_loss_oversample.append(
                    criterion_training_01(train_outputs_positive, train_outputs_unlabeled, prior_prime).item())

            # training_sigmoid_losses.append(np.mean(training_sigmoid_loss))
            training_01_losses.append(np.mean(training_01_loss))
            training_01_losses_oversample.append(np.mean(training_01_loss_oversample))

            # test loss
            # test loss
            test_01_loss = []
            test_precion = []
            test_recall = []
            test_F1 = []
            test_01_loss_oversample = []
            auc_loss = []
            for (test_positive, test_negative) in zip(testloader_positive, testloader_negative):
                # Get positive data and negative data
                positive_instances, positive_labels = test_positive
                negative_instances, negative_labels = test_negative
                positive_instances = positive_instances.to(device)
                negative_instances = negative_instances.to(device)
                test_outputs_positive = model(positive_instances)
                test_outputs_negative = model(negative_instances)
                test_output = torch.cat((test_outputs_positive, test_outputs_negative), 0)
                test_target = torch.cat((positive_labels, negative_labels), 0)

                # test 01 loss
                test_01_loss.append(criterion_test_01(test_outputs_positive, test_outputs_negative, prior).item())
                test_01_loss_oversample.append(
                    criterion_test_01(test_outputs_positive, test_outputs_negative, prior_prime).item())
                # Others
                test_precion.append(criterion_precision(test_outputs_positive, test_outputs_negative, prior))
                test_recall.append(criterion_recall(test_outputs_positive, test_outputs_negative, prior))
                test_F1.append(criterion_F1(test_outputs_positive, test_outputs_negative, prior))

                # output
                auc_loss.append(criterion_Auc(test_output, test_target).item())

            test_01_losses.append(np.mean(test_01_loss))
            test_01_losses_oversample.append(np.mean(test_01_loss_oversample))
            test_precions.append(np.mean(test_precion))
            test_recalls.append(np.mean(test_recall))
            test_F1s.append(np.mean(test_F1))
            Auc_losses.append(np.mean(auc_loss))

        model.train()

        print(epoch)
        print(test_F1s)
        print(Auc_losses)
        print(test_01_losses)


        savename = 'dummy-folder/nnpu_sigmoid_result_pos_%i_lr_%i_wd_%i_batches_%i_trial_%i.mat' %(num_initial_pos, learning_rate, weight_decay, num_batches, seed)
        scio.savemat(savename,
                     {'prior': prior,
                      'running_losses': running_losses,
                      'training_sigmoid_losses': training_sigmoid_losses,
                      'training_01_losses': training_01_losses,
                      'test_01_losses': test_01_losses,
                      'training_01_losses_oversample': training_01_losses_oversample,
                      'test_01_losses_oversample': test_01_losses_oversample,
                      'test_precisions': test_precions,
                      'test_recalls': test_recalls,
                      'test_F1s': test_F1s,
                      'test_AUCs': Auc_losses})

