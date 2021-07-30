import argparse

from torch.backends import cudnn

from data_loader import get_loader
from utils.other_util import *
from solver import Solver


def main(config):
    cudnn.benchmark = True
    # 结果保存地址
    config.result_path = os.path.join(config.result_path,
                                      config.Task_name + str(config.fold_K) + '_' + str(config.fold_idx))
    print(config.result_path)
    # 模型以及训练过程参数保存地址
    config.model_path = os.path.join(config.result_path, 'models')
    config.log_dir = os.path.join(config.result_path, 'logs')
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
        os.makedirs(config.model_path)
        os.makedirs(config.log_dir)
        os.makedirs(os.path.join(config.result_path, 'images'))
    # 显卡选择/不并行
    if not config.DataParallel:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_idx)

    print(config)
    # 设置保存txt
    f = open(os.path.join(config.result_path, 'config.txt'), 'w')
    for key in config.__dict__:
        print('%s: %s' % (key, config.__getattribute__(key)), file=f)
    f.close()

    # 获取交叉验证的数据划分
    train, test = get_fold_filelist(config.csv_file, K=config.fold_K, fold=config.fold_idx)

    train_list = [config.filepath_img + sep + i[0] for i in train]
    train_list_GT = [config.filepath_mask + sep + i[0] for i in train]
    train_pix_hc = [[i[2], i[3]] for i in train]

    test_list = [config.filepath_img + sep + i[0] for i in test]
    test_list_GT = [config.filepath_mask + sep + i[0] for i in test]
    test_pix_hc = [[i[2], i[3]] for i in test]

    # 测试集即验证集
    valid_list = test_list
    valid_list_GT = test_list_GT
    valid_pix_hc = test_pix_hc

    config.train_list = train_list
    config.test_list = test_list
    config.valid_list = valid_list
    config.train_pix_hc = train_pix_hc
    config.test_pix_hc = test_pix_hc
    config.valid_pix_hc = valid_pix_hc

    train_loader = get_loader(pix_hc_list=train_pix_hc,
                              GT_list=train_list_GT,
                              image_list=train_list,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)

    valid_loader = get_loader(pix_hc_list=valid_pix_hc,
                              GT_list=valid_list_GT,
                              image_list=valid_list,
                              image_size=config.image_size,
                              batch_size=config.batch_size_test,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0.)

    test_loader = get_loader(pix_hc_list=test_pix_hc,
                             GT_list=test_list_GT,
                             image_list=test_list,
                             image_size=config.image_size,
                             batch_size=config.batch_size_test,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        unet_path = os.path.join(config.model_path, 'best_unet_score.pkl')
        acc, SE, SP, PC, DC, IOU, HCD = solver.test(mode='test', unet_path=unet_path)
        print('[Testing]    Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f, HCD: %.4f' % (
            acc, SE, SP, PC, DC, IOU, HCD))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=256)  # 网络输入img的size, 即输入会被强制resize到这个大小

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=290)

    # 普通的lr阶梯式下降，没有使用
    # parser.add_argument('--num_epochs_decay', type=int, default=60)  # decay开始的最小epoch数
    # parser.add_argument('--decay_ratio', type=float, default=0.01)  # 0~1,每次decay到1*ratio
    # parser.add_argument('--decay_step', type=int, default=60)  # epoch

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_test', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=4)

    # 设置学习率
    parser.add_argument('--lr', type=float, default=1e-4)  # 初始or最大学习率(单用lovz且多gpu的时候,lr貌似要大一些才可收敛)
    parser.add_argument('--lr_low', type=float, default=1e-12)  # 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)

    parser.add_argument('--lr_warm_epoch', type=int, default=10)  # warmup的epoch数,一般就是5~20,为0或False则不使用
    parser.add_argument('--lr_cos_epoch', type=int, default=250)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用

    # optimizer param
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

    parser.add_argument('--augmentation_prob', type=float, default=0.5)  # 扩增几率

    parser.add_argument('--save_model_step', type=int, default=300)
    parser.add_argument('--val_step', type=int, default=5)

    # misc
    parser.add_argument('--mode', type=str, default='test', help='train/test')
    parser.add_argument('--Task_name', type=str, default='HC_regy64-', help='DIR name,Task name')
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--DataParallel', type=bool, default=False)

    # data-parameters
    parser.add_argument('--filepath_img', type=str, default='/data/medai05/EDAN2021/data/training_set/images')
    parser.add_argument('--filepath_mask', type=str, default='/data/medai05/EDAN2021/data/training_set/labels_pre')
    parser.add_argument('--csv_file', type=str, default='/data/medai05/EDAN2021/data/train.csv')    # 用于分折
    parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')
    parser.add_argument('--fold_idx', type=int, default=1)

    # result&save
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--save_detail_result', type=bool, default=True)
    parser.add_argument('--save_image', type=bool, default=True)  # 训练过程中观察图像和结果

    # more param
    parser.add_argument('--test_flag', type=bool, default=False)  # 训练过程中是否测试,不测试会节省很多时间,事实上验证集就是测试集

    config = parser.parse_args()
    main(config)
