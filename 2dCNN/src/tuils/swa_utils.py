from tqdm import tqdm

import torch
import torch.nn.functional as F


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.zeros_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input_dict in tqdm(loader):
        inputs, _ = input_dict
        inputs = inputs.cuda(async=True)
        input_var = torch.autograd.Variable(inputs)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))

def detach_params(model):
    for param in model.parameters():
        param.detach_()

    return model

def evaluate(loader, model):
    model.eval()

    out_pred = torch.FloatTensor().cuda()
    out_gt = torch.FloatTensor().cuda()

    for input_dict in loader:
        inputs, targets = input_dict
        inputs = inputs.cuda()
        targets = targets.cuda().float()

        logits = model(inputs)
        probabilities = torch.sigmoid(logits)

        out_pred = torch.cat((out_pred, probabilities), 0)
        out_gt = torch.cat((out_gt, targets), 0)

    eval_metric_bundle = search_f2(out_pred, out_gt)
    print('===> Best', eval_metric_bundle)
    
    return eval_metric_bundle