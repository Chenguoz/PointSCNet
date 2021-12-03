import re
from matplotlib import *
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import make_interp_spline

pathlist = [
    r'C:\Users\LENOVO\Desktop\Pointnet\Pointnet_Pointnet2_pytorch-master\log\classification\test10\logs\pointnet_srn_cls_msg.txt']


def read_log(log_path_list, logs, train_acc=True, test_acc=True, loss=True):

    fig = plt.figure(num=1, figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # 绘画处理
    ax1.set_xlabel('epoch')
    ax1.set_title('accuracy')
    ax1.axis([0, 200, 0.9, 0.95])
    ax2.set_xlabel('epoch')
    ax2.set_title('total_loss')
    ax2.axis([0, 100, 0, 300])
    for idx, path in enumerate(log_path_list):
        log_dir = path
        print(path)
        with open(log_dir, "r", encoding="utf-8") as f:
            content = f.read()

        epoch = re.findall(r'Epoch \d[0-9]*', content, re.M)
        Train_Instance_Accuracy = re.findall(r'Train Instance Accuracy: .*', content, re.M)
        if loss:
            Total_loss = re.findall(r'Total loss: .*', content, re.M)
        Test_Instance_Accuracy = re.findall(r'Test Instance Accuracy: .*,', content, re.M)

        data_length = min([len(epoch), len(Train_Instance_Accuracy), len(Test_Instance_Accuracy)])
        for i in range(data_length):
            epoch[i] = int(epoch[i].strip('Epoch '))
            Train_Instance_Accuracy[i] = float(Train_Instance_Accuracy[i].strip('Train Instance Accuracy: '))
            if loss:
                Total_loss[i] = float(Total_loss[i].strip('Total loss: '))
            Test_Instance_Accuracy[i] = float(Test_Instance_Accuracy[i].strip('Test Instance Accuracy: ,'))
        plt.rcParams["font.family"] = "SimHei"
        if loss:
            total_loss_x = epoch[:data_length]
            total_loss_y = Total_loss[:data_length]

        test_accuracy_x = epoch[:data_length]
        test_accuracy_y = Test_Instance_Accuracy[:data_length]

        train_accuracy_x = epoch[:data_length]
        train_accuracy_y = Train_Instance_Accuracy[:data_length]
        if test_acc:
            # test_accuracy_x = np.array(test_accuracy_x)
            # test_accuracy_y = np.array(test_accuracy_x)
            # x_new = np.linspace(test_accuracy_x.min(), test_accuracy_x.max(),
            #                     1000)  # 1000 represents number of points to make between T.min and T.max
            # y_smooth = make_interp_spline(test_accuracy_x, test_accuracy_y)(x_new)
            ax1.plot(test_accuracy_x, test_accuracy_y, label='test_accuracy_{}'.format(logs[idx]))
            # ax1.plot(x_new, y_smooth, label='test_accuracy_{}'.format(logs[idx]))

            ax1.legend(loc='best', labelspacing=1, handlelength=4, fontsize=14, shadow=True)
        if train_acc:
            ax1.plot(train_accuracy_x, train_accuracy_y, label='train_accuracy_{}'.format(logs[idx]))
            ax1.legend(loc='best', labelspacing=1, handlelength=4, fontsize=14, shadow=True)
        if loss:
            ax2.plot(total_loss_x, total_loss_y, label='loss_{}'.format(logs[idx]))
            ax2.legend(loc='best', labelspacing=1, handlelength=4, fontsize=14, shadow=True)
    plt.show()


def read_all_paths(log_dir):
    class_dirs = os.listdir(log_dir)
    paths = [os.path.join(log_dir, i) + r'\logs' for i in class_dirs]
    all_paths = [os.path.join(i, os.listdir(i)[0]) for i in paths]
    return all_paths


def read_log_path(log_name, classfication=True):
    log_dir = []
    if classfication:
        for i in range(len(log_name)):
            log_dir.append('log/classification/' + log_name[i] + "/logs/" + \
                           os.listdir('log/classification/' + log_name[i] + r'\logs')[0])
    return log_dir


if __name__ == "__main__":
    logs = ['Ours', 'Pointnet++']
    lists = read_log_path(logs)
    read_log(lists, logs, train_acc=False, loss=True)

    # 提取数据
    # log_dir = r'C:\Users\LENOVO\Desktop\Pointnet\Pointnet_Pointnet2_pytorch-master\log\classification\test10\logs\pointnet_srn_cls_msg.txt'
    #
    # with open(log_dir, "r", encoding="utf-8") as f:
    #     content = f.read()
    # epoch = re.findall(r'Epoch \d[0-9]*', content, re.M)
    # Train_Instance_Accuracy = re.findall(r'Train Instance Accuracy: .*', content, re.M)
    # Total_loss = re.findall(r'Total loss: .*', content, re.M)
    # Test_Instance_Accuracy = re.findall(r'Test Instance Accuracy: .*,', content, re.M)
    # for i in range(len(epoch)):
    #     epoch[i] = int(epoch[i].strip('Epoch '))
    #     Train_Instance_Accuracy[i] = float(Train_Instance_Accuracy[i].strip('Train Instance Accuracy: '))
    #     Total_loss[i] = float(Total_loss[i].strip('Total loss: '))
    #     Test_Instance_Accuracy[i] = float(Test_Instance_Accuracy[i].strip('Test Instance Accuracy: ,'))
    #
    # plt.rcParams["font.family"] = "SimHei"
    # total_loss_x = epoch
    # total_loss_y = Total_loss
    # test_accuracy_x = epoch
    # test_accuracy_y = Test_Instance_Accuracy
    # train_accuracy_x = epoch
    # train_accuracy_y = Train_Instance_Accuracy
    #
    # fig = plt.figure(num=1, figsize=(4, 4))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # # 绘画处理
    # ax1.set_xlabel('epoch')
    # ax1.set_title('accuracy')
    # ax2.set_xlabel('epoch')
    # ax2.set_title('total_loss')
    #
    # ax1.plot(test_accuracy_x, test_accuracy_y, "r-", label='test_accuracy')
    #
    # ax1.legend(loc='best', labelspacing=1, handlelength=4, fontsize=14, shadow=True)
    # ax1.plot(train_accuracy_x, train_accuracy_y, "b-", label='train_accuracy')
    # ax1.legend(loc='best', labelspacing=1, handlelength=4, fontsize=14, shadow=True)
    #
    # ax2.plot(total_loss_x, total_loss_y, "c-")
    # plt.show()
