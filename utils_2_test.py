import datetime
import os
from collections import deque
import mxnet
from mxnet import gluon, nd
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm


def test(metric, ctx, net, val_data, num_views=1, num_class=None, if_caps=False, use_viewpoints=False):
    assert num_views >= 1, "'num_views' should be greater or equal to 1"
    metric.reset()
    iiiii = 0
    true_labels = []
    predict_labels = []

    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    val_data = tqdm(val_data)
    for data, depth_1, depth_2, label, *rest in val_data:
        position_1 = depth_1
        position_2 = depth_2
        if use_viewpoints:
            _, shuffle_idx, _ = rest
        if data.shape[0] == 1:
            Xs = [data.as_in_context(ctx[0])]
            Ys = [label.as_in_context(ctx[0])]
            position_1 = [position_1.as_in_context(ctx[0])]
            position_2 = [position_2.as_in_context(ctx[0])]
            if use_viewpoints:
                IDs = [shuffle_idx.as_in_context(ctx[0])]
        else:
            Xs = gluon.utils.split_and_load(data,
                                            ctx_list=ctx, batch_axis=0, even_split=False)
            Ys = gluon.utils.split_and_load(label,
                                            ctx_list=ctx, batch_axis=0, even_split=False)
            position_1 = gluon.utils.split_and_load(position_1,
                                                    ctx_list=ctx, batch_axis=0, even_split=False)
            position_2 = gluon.utils.split_and_load(position_2,
                                                    ctx_list=ctx, batch_axis=0, even_split=False)
            if use_viewpoints:
                IDs = gluon.utils.split_and_load(shuffle_idx,
                                                 ctx_list=ctx, batch_axis=0, even_split=False)
        if not use_viewpoints:
            if if_caps:
                outputs = [net(X)[1].squeeze(axis=-1) for X in Xs]
            else:
                if num_views > 1:
                    outputs = [net(X).reshape(-1, num_views, num_class).mean(axis=1) for X in Xs]
                else:
                    outputs = []
                    for x, y, z, w in zip(Xs, Ys, position_1, position_2):
                        iiiii += 1
                        out = net(x, z, w)
                        outputs.append(out)
                        y1 = int(y.asnumpy())
                        true_labels.append(y1)
                        out1 = list(np.squeeze(out.asnumpy()))
                        out1 = out1.index(max(out1))
                        predict_labels.append(out1)
        else:
            if num_views > 1:
                outputs = [net(X, ID)[0].reshape(-1, num_views, num_class).mean(axis=1) for X, ID in zip(Xs, IDs)]
            else:
                outputs = [net(X, ID)[0] for X, ID in zip(Xs, IDs)]
        metric.update(Ys, outputs)
    cal_class_acc(true_labels, predict_labels)
    return metric.get()


def cal_class_acc(true_labels, predict_labels):
    nums = 15
    true_, pre_ = [0] * nums, [0] * nums
    for i in range(len(true_labels)):
        true_[true_labels[i]] += 1
        if true_labels[i] == predict_labels[i]:
            pre_[true_labels[i]] += 1
    a = 0
    for i in range(len(true_)):
        if true_[i] != 0:
            temp = round(pre_[i] / true_[i], 3)
            a += temp
            print(i, ':', temp)
    print('over_all:', round(a / nums, 3))


def get_format_time_string(time_interval):
    h, remainder = divmod((time_interval).seconds, 3600)
    m, s = divmod(remainder, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def get_confusion_matrix(net, val_data, ctx, num_views=1, num_class=None):
    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    y_preds, y_trues = [], []
    for data, label in val_data:
        if data.shape[0] == 1:
            Xs = [data.as_in_context(ctx[0])]
        else:
            Xs = gluon.utils.split_and_load(data,
                                            ctx_list=ctx, batch_axis=0, even_split=False)

        if num_views > 1:
            outputs = [net(X).reshape(-1, num_views, num_class).mean(axis=1) for X in Xs]
        else:
            outputs = [net(X) for X in Xs]
        output_labels = [out.argmax(axis=1) for out in outputs]
        y_preds.append(nd.concat(*output_labels, dim=0).astype('uint8').asnumpy())
        y_trues.append(label.asnumpy())
    return confusion_matrix(np.concatenate(y_trues, axis=None), np.concatenate(y_preds, axis=None))


def get_view_sequences(num_views):
    s = deque(range(num_views))
    seqs = []
    for i in range(len(s)):
        s.rotate(1)
        seqs.append(list(s))
    s_r = deque(range(num_views - 1, -1, -1))
    for i in range(len(s_r)):
        s_r.rotate(1)
        seqs.append(list(s_r))
    return seqs


def log_string(log_out, out_str):
    log_out.write(out_str + '\n')
    log_out.flush()


def smooth(label, classes, eta=0.1):
    ind = label.astype('int')
    res = nd.zeros((ind.shape[0], classes), ctx=label.context)
    res += eta / classes
    res[nd.arange(ind.shape[0], ctx=label.context), ind] = 1 - eta + eta / classes
    return res


def save_checkpoint(net, current_epoch, checkpoint_prefix):
    net.save_parameters(os.path.join(checkpoint_prefix, 'Epoch%s.params' % current_epoch))


def train(net, train_data, test_data, loss_fun, kvstore, log_out, checkpoint_prefix, train_args):
    trainer_dict = {'learning_rate': train_args.lr, 'wd': train_args.wd}
    if train_args.optimizer == 'sgd':
        trainer_dict['momentum'] = 0.9
    metric = mxnet.metric.Accuracy()

    ctx = [mxnet.gpu(gpu_id) for gpu_id in train_args.gpu]

    log_string(log_out, str(datetime.datetime.now()))
    log_string(log_out, net.get_info())
    log_string(log_out, str(train_args))
    print('start training on %s' % train_args.dataset_path)

    for epoch in range(train_args.from_epoch, train_args.epoch):
        _, train_acc = metric.get()
        if train_args.multi_output:
            _, test_acc = test(metric, ctx, net, test_data, num_views=train_args.num_views,
                               num_class=train_args.num_classes)
        else:
            _, test_acc = test(metric, ctx, net, test_data)
        epoch_str = "Valid acc: %f" % (test_acc)
        print(epoch_str)
        log_string(log_out, epoch_str)
