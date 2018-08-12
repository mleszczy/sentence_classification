import os
import sys
import argparse
import time
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import dataloader
import modules

from utils import *
import logging

logger = logging.getLogger(__name__)

FORMAT = '%(levelname)s|%(asctime)s|%(name)s|line_num:%(lineno)d| %(message)s'

class Model(nn.Module):
    def __init__(self, args, emb_layer, nclasses=2):
        super(Model, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args.dropout)
        self.emb_layer = emb_layer
        if args.cnn:
            self.encoder = modules.CNN_Text(
                emb_layer.n_d,
                widths = [3,4,5]
            )
            d_out = 300
        elif args.lstm:
            self.encoder = nn.LSTM(
                emb_layer.n_d,
                args.d,
                args.depth,
                dropout = args.dropout,
            )
            d_out = args.d
        elif args.la:
            d_out = emb_layer.n_d
        self.out = nn.Linear(d_out, nclasses)

    def forward(self, input):
        if self.args.cnn:
            input = input.t()
        emb = self.emb_layer(input)
        if not self.args.la:
            emb = self.drop(emb)
        if self.args.cnn:
            output = self.encoder(emb)
        elif self.args.lstm:
            output, hidden = self.encoder(emb)
            output = output[-1]
        else:
            output = emb.sum(dim=0) / emb.size()[0]
        if not self.args.la:
            output = self.drop(output)
        return self.out(output)

def loss_kd(outputs, labels, teacher_outputs, args):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = args.alpha
    T = args.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

def eval_model(niter, model, valid_x, valid_y, pred_file=None):
    model.eval()
    N = len(valid_x)
    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0.0
    total_loss = 0.0
    preds = []
    for x, y in zip(valid_x, valid_y):
        x, y = Variable(x, volatile=True), Variable(y)
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.data[0]*x.size(1)
        pred = output.data.max(1)[1]
        correct += pred.eq(y.data).cpu().sum().item()
        cnt += y.numel()
        preds.extend(list(pred.cpu().numpy()))

    if pred_file is not None:
        with open(pred_file, 'wb') as outfile:
            pickle.dump(preds, outfile)

    model.train()
    return 1.0-correct/cnt

def train_model(epoch, model, optimizer,
        train_x, train_y, valid_x, valid_y,
        test_x, test_y,
        best_valid, test_err,
        teacher_models,
        save_mdl=None,
        pred_file=None):

    model.train()
    args = model.args
    N = len(train_x)
    niter = epoch*len(train_x)
    # criterion = nn.CrossEntropyLoss()
    criterion = loss_kd

    cnt = 0
    for x, y in zip(train_x, train_y):
        niter += 1
        cnt += 1
        model.zero_grad()
        x, y = Variable(x), Variable(y)
        output = model(x)
        teacher_outputs = None
        for teacher_model in teacher_models:
            teacher_model.eval()
            teacher_output = teacher_model(x).data
            if teacher_outputs is not None:
                teacher_outputs *= teacher_output
            else:
                teacher_outputs = teacher_output
        teacher_outputs = teacher_outputs**(1.0/len(teacher_outputs))
        loss = criterion(output, y, Variable(teacher_outputs), args)
        loss.backward()
        optimizer.step()

    valid_err = eval_model(niter, model, valid_x, valid_y)

    logger.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_err={:.6f}".format(
        epoch, niter,
        optimizer.param_groups[0]['lr'],
        loss.data[0],
        valid_err
    ))

    if valid_err < best_valid:
        best_valid = valid_err
        test_err = eval_model(niter, model, test_x, test_y, pred_file)
        # Save model
        if save_mdl is not None:
            torch.save(model, save_mdl)

    return best_valid, test_err

def main(args):
    if args.dataset == 'mr':
        data, label = dataloader.read_MR(args.path, seed=args.data_seed)
    elif args.dataset == 'subj':
        data, label = dataloader.read_SUBJ(args.path, seed=args.data_seed)
    elif args.dataset == 'cr':
        data, label = dataloader.read_CR(args.path, seed=args.data_seed)
    elif args.dataset == 'mpqa':
        data, label = dataloader.read_MPQA(args.path, seed=args.data_seed)
    elif args.dataset == 'trec':
        train_x, train_y, test_x, test_y = dataloader.read_TREC(args.path, seed=args.data_seed)
        data = train_x + test_x
        label = None
    elif args.dataset == 'sst':
        train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_SST(args.path, seed=args.data_seed)
        data = train_x + valid_x + test_x
        label = None
    else:
        raise Exception("unknown dataset: {}".format(args.dataset))

    emb_layer = modules.EmbeddingLayer(
        args.d, data,
        embs = dataloader.load_embedding(args.embedding)
    )

    if args.dataset == 'trec':
        train_x, train_y, valid_x, valid_y = dataloader.cv_split2(
            train_x, train_y,
            nfold = 10,
            valid_id = args.cv
        )
    elif args.dataset != 'sst':
        train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.cv_split(
            data, label,
            nfold = 10,
            test_id = args.cv
        )

    nclasses = max(train_y)+1

    train_x, train_y = dataloader.create_batches(
        train_x, train_y,
        args.batch_size,
        emb_layer.word2id,
        sort = args.dataset == 'sst'
    )
    valid_x, valid_y = dataloader.create_batches(
        valid_x, valid_y,
        args.batch_size,
        emb_layer.word2id,
        sort = args.dataset == 'sst'
    )
    test_x, test_y = dataloader.create_batches(
        test_x, test_y,
        args.batch_size,
        emb_layer.word2id,
        sort = args.dataset == 'sst'
    )


    # Set random seed for torch
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed(args.model_seed)

    # Set random seed for numpy
    np.random.seed(seed=args.model_seed)

    if args.load_mdl is None:
        model = Model(args, emb_layer, nclasses).cuda()
    else:
        # Note: this will overwrite all parameters
        model = torch.load(args.load_mdl).cuda()

    teacher_models = [torch.load(_).cuda() for _ in args.teacher_model_path.split(',')]

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr = args.lr
    )

    best_valid = 1e+8
    test_err = 1e+8
    for epoch in range(args.max_epoch):
        best_valid, test_err = train_model(epoch, model, optimizer,
            train_x, train_y,
            valid_x, valid_y,
            test_x, test_y,
            best_valid, test_err,
            teacher_models,
            args.save_mdl,
            args.out
        )
        if args.lr_decay>0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

    logger.info("=" * 40)
    logger.info("best_valid: {:.6f}".format(
        best_valid
    ))
    logger.info("test_err: {:.6f}".format(
        test_err
    ))
    logger.info("=" * 40)


if __name__ == "__main__":

    # Set logging
    logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.INFO)

    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    
    argparser.add_argument("--dataset", type=str, default="mr", help="which dataset")
    argparser.add_argument("--path", type=str, required=True, help="path to corpus directory")
    argparser.add_argument("--data_seed", type=int, default=1234)
    argparser.add_argument("--embedding", type=str, required=True, help="word vectors")

    argparser.add_argument("--teacher_model_path", type=str, default="lstm", help="teacher model path (delimiter=,)")

    argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")
    argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
    argparser.add_argument("--la", action='store_true', help="whether to use la")

    # argparser.add_argument("--student_model", type=str, default="lstm", help="student model")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--d", type=int, default=128)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--depth", type=int, default=2)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0.0)
    argparser.add_argument("--cv", type=int, default=0)
    argparser.add_argument("--model_seed", type=int, default=1234)
    argparser.add_argument("--out", type=str, help="Save predictions to this file.")

    argparser.add_argument("--save_mdl", type=str, default=None, help="Save model to this file.")
    argparser.add_argument("--load_mdl", type=str, default=None, help="Load model from this file.")
    argparser.add_argument("--alpha", type=float, default=0.9)
    argparser.add_argument("--temperature", type=float, default=5.0)
    args = argparser.parse_args()

    # Dump command line arguments
    logger.info("Machine: " + os.uname()[1])
    logger.info("CMD: python " +  " ".join(sys.argv))
    print_key_pairs(args.__dict__.items(), title="Command Line Args", print_function=logger.info)
    # print (args)
    main(args)
