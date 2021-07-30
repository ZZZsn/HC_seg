from sklearn.model_selection import KFold
import os
import warnings
import random
import numpy as np
import csv

warnings.filterwarnings('ignore')
sep = os.sep
filesep = sep  # 设置分隔符


def char_color(s, front=50, word=32):
    """
    # 改变字符串颜色的函数
    :param s: 
    :param front: 
    :param word: 
    :return: 
    """
    new_char = "\033[0;" + str(int(word)) + ";" + str(int(front)) + "m" + s + "\033[0m"
    return new_char


def array_shuffle(x, axis=0, random_state=2020):
    """
    对多维度数组，在任意轴打乱顺序
    :param x: ndarray
    :param axis: 打乱的轴
    :return:打乱后的数组
    """
    new_index = list(range(x.shape[axis]))
    random.seed(random_state)
    random.shuffle(new_index)
    x_new = np.transpose(x, ([axis] + [i for i in list(range(len(x.shape))) if i is not axis]))
    x_new = x_new[new_index][:]
    new_dim = list(np.array(range(axis)) + 1) + [0] + list(np.array(range(len(x.shape) - axis - 1)) + axis + 1)
    x_new = np.transpose(x_new, tuple(new_dim))
    return x_new


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', content=None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if content:
        print('\r%s |%s| %s%% %s %s' % (prefix, bar, percent, suffix, content), end=' ')
    else:
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=' ')

    # Print New Line on Complete
    if iteration == total:
        print()


def get_filelist_frompath(filepath, expname, sample_id=None):
    """
    读取文件夹中带有固定扩展名的文件
    :param filepath:
    :param expname: 扩展名，如'h5','PNG'
    :param sample_id: 可以只读取固定患者id的图片
    :return: 文件路径list
    """
    file_name = os.listdir(filepath)
    file_List = []
    if sample_id is not None:
        for file in file_name:
            if file.endswith('.' + expname):
                id = int(file.split('.')[0])
                if id in sample_id:
                    file_List.append(os.path.join(filepath, file))
    else:
        for file in file_name:
            if file.endswith('.' + expname):
                file_List.append(os.path.join(filepath, file))
    return file_List


def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines


def get_fold_filelist(csv_file, K=5, fold=1, random_state=2021):
    """
    获取分折结果的API（基于size分3层的类别平衡分折）
    :param csv_file: 带有ID、size的文件
    :param K: 分折折数
    :param fold: 返回第几折,从1开始
    :param random_state: 随机数种子, 固定后每次实验分折相同(注意,sklearn切换版本可能会导致相同随机数种子产生不同分折结果)
    :param validation: 是否需要验证集（从训练集随机抽取部分数据当作验证集）
    :param validation_r: 抽取出验证集占训练集的比例
    :return: train和test的list，带有label和size
    """

    CTcsvlines = readCsv(csv_file)
    header = CTcsvlines[0]
    print('header', header)
    nodules = CTcsvlines[1:]

    # 提取label_num的三分点
    sizeall = [i[1] for i in nodules]
    sizeall.sort()
    low_mid_thre = sizeall[int(len(sizeall) * 1 / 3)]
    mid_high_thre = sizeall[int(len(sizeall) * 2 / 3)]

    # 根据size三分位数分为low，mid，high三组
    low_size_list = [i for i in nodules if i[1] < low_mid_thre]
    mid_size_list = [i for i in nodules if mid_high_thre > i[1] >= low_mid_thre]
    high_size_list = [i for i in nodules if i[1] >= mid_high_thre]

    low_fold_train = []
    low_fold_test = []

    mid_fold_train = []
    mid_fold_test = []

    high_fold_train = []
    high_fold_test = []

    sfolder = KFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(low_size_list):
        low_fold_train.append([low_size_list[i] for i in train])
        low_fold_test.append([low_size_list[i] for i in test])

    sfolder = KFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(mid_size_list):
        mid_fold_train.append([mid_size_list[i] for i in train])
        mid_fold_test.append([mid_size_list[i] for i in test])

    sfolder = KFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(high_size_list):
        high_fold_train.append([high_size_list[i] for i in train])
        high_fold_test.append([high_size_list[i] for i in test])

    train_set = low_fold_train[fold - 1] + mid_fold_train[fold - 1] + high_fold_train[fold - 1]
    test_set = low_fold_test[fold - 1] + mid_fold_test[fold - 1] + high_fold_test[fold - 1]
    return [train_set, test_set]


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
