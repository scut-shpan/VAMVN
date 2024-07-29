import argparse
import datetime
import os
import sys
from pathlib import Path
import mxnet
from mxnet import gluon
import importlib
import utils_2_test as utils
from datahelper import MultiViewImageDataset
from mxnet.gluon.data.vision.transforms import Compose, ToTensor, Normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('MV')

    parser.add_argument('--model', type=str, default='models.model_resnet_fusion_test', help='name of the model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--batch_update_period', type=int, default=1, help='batch size in training')
    parser.add_argument('--epoch', default=1, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='specify gpu device')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_period', type=int, default=20, help='learning rate decay period')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--dataset_path', type=str, default=r'./dataset/ScanObjectNN/shaded',
                        help='path to the dataset')
    parser.add_argument('--checkpoint', type=str,
                        default='./pretrainedParams/Epoch_best.params', help='location of the checkpoint')
    parser.add_argument('--pretrained_cnn', type=str, default=None, help='location of the 2d pretrained_cnn')
    parser.add_argument('--output_lr_mult', type=float, default=10, help='lr multiplier for output layer')
    parser.add_argument('--num_views', type=int, default=20, help='number of views')
    parser.add_argument('--num_classes', type=int, default=15, help='number of classes')
    parser.add_argument('--use_viewpoints', type=int, default=0, help='number of classes')
    parser.add_argument('--pretrained', default=True, action='store_false', help='whether to use the pretrained model')
    parser.add_argument('--label_smoothing', default=False, action='store_true',
                        help='whether to use the label smoothing')
    parser.add_argument('--multi_output', default=False, action='store_true',
                        help='whether to output result for each view')
    parser.add_argument('--disable_sort', default=False, action='store_true',
                        help='whether to disable sorting,for rank pooling')
    parser.add_argument('--shuffle', default=False, action='store_true', help='whether to shuffle the view sequences')
    parser.add_argument('--from_epoch', default=0, type=int, help='start from epoch(for training from checkpoint)')
    parser.add_argument('--pooling_type', default='rank', type=str, help='pooling type for MVCNN')
    parser.add_argument('--use_sample_weights', default=False, action='store_true',
                        help='whether to use sample_weights for cp loss')
    parser.add_argument('--feature_length', type=int, default=2048,
                        help='the length of each view on view feature level')
    parser.add_argument('--resnet_parms_path', help='the path to resnet paramters')

    return parser.parse_args()


def main(args):
    ''' create dir '''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = Path('./experiment/checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = Path('./experiment/logs/')
    log_dir.mkdir(exist_ok=True)

    ctx = mxnet.gpu(args.gpu)

    model = importlib.import_module(args.model)
    net = model.get_model(args)

    '''Setup loss function'''
    loss_fun = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)

    '''Loading dataset'''
    train_ds = MultiViewImageDataset(os.path.join(args.dataset_path, 'train'), args.num_views,
                                     transform=Compose([
                                         ToTensor(),
                                         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]))
    test_ds = MultiViewImageDataset(os.path.join(args.dataset_path, 'test'), args.num_views,
                                    transform=Compose([
                                        ToTensor(),
                                        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]))
    loader = gluon.data.DataLoader
    train_data = loader(train_ds, args.batch_size, shuffle=True, last_batch='keep', num_workers=0)
    test_data = loader(test_ds, args.batch_size, shuffle=False, last_batch='keep', num_workers=0)

    current_time = datetime.datetime.now()
    time_str = '%d-%d-%d--%d-%d-%d' % (
        current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute,
        current_time.second)
    log_filename = time_str + '.txt'
    checkpoint_name = 'checkpoint_' + time_str
    checkpoint_dir = Path(os.path.join(checkpoints_dir, checkpoint_name))
    checkpoint_dir.mkdir(exist_ok=True)

    with open(os.path.join(log_dir, log_filename, ), 'w') as log_out:
        try:
            kv = mxnet.kv.create('device')
            utils.log_string(log_out, sys.argv[0])
            utils.train(net, train_data, test_data, loss_fun, kv, log_out, str(checkpoint_dir), args)
        except Exception as e:
            raise e


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    print(str(args))
    main(args)
