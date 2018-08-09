import os
import sys
import argparse
import time
import pickle
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# import cuda_functional as MF
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

def eval_model(model, valid_x, valid_y, pred_file=None, prob_file=None):
    model.eval()
    N = len(valid_x)
    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0.0
    total_loss = 0.0
    preds = []
    probs = []
    for x, y in zip(valid_x, valid_y):
        x, y = Variable(x, volatile=True), Variable(y)
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.data[0]*x.size(1)
        pred = output.data.max(1)[1]
        correct += pred.eq(y.data).cpu().sum().item()
        cnt += y.numel()
        preds += pred.cpu().numpy().tolist()
        probs += output.data.cpu().numpy().tolist()

    if pred_file is not None:
        with open(pred_file, 'wb') as outfile:
            pickle.dump(preds, outfile)

    if prob_file is not None:
         with open(prob_file, 'wb') as outfile:
            pickle.dump(probs, outfile)

    model.train()
    return 1.0-correct/cnt

def train_model(epoch, model, optimizer,
        train_x, train_y, valid_x, valid_y,
        test_x, test_y,
        best_valid, test_err,
        pred_file=None, prob_file=None):

    model.train()
    args = model.args
    N = len(train_x)
    niter = epoch*len(train_x)
    criterion = nn.CrossEntropyLoss()

    cnt = 0
    for x, y in zip(train_x, train_y):
        niter += 1
        cnt += 1
        model.zero_grad()
        x, y = Variable(x), Variable(y)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    valid_err = eval_model(model, valid_x, valid_y)

    logger.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_err={:.6f}".format(
        epoch, niter,
        optimizer.param_groups[0]['lr'],
        loss.data[0],
        valid_err
    ))

    if valid_err < best_valid:
        best_valid = valid_err
        test_err = eval_model(model, test_x, test_y, pred_file, prob_file)
    return best_valid, test_err

def cyclic_lr(initial_lr, iteration, epoch_per_cycle):
    return initial_lr * (math.cos(math.pi * iteration / epoch_per_cycle) + 1) / 2

def main(args):
    if args.dataset == 'mr':
        data, label = dataloader.read_MR(args.path, seed=args.seed)
    elif args.dataset == 'subj':
        data, label = dataloader.read_SUBJ(args.path, seed=args.seed)
    elif args.dataset == 'cr':
        data, label = dataloader.read_CR(args.path, seed=args.seed)
    elif args.dataset == 'mpqa':
        data, label = dataloader.read_MPQA(args.path, seed=args.seed)
    elif args.dataset == 'trec':
        train_x, train_y, test_x, test_y = dataloader.read_TREC(args.path, seed=args.seed)
        data = train_x + test_x
        label = None
    elif args.dataset == 'sst':
        train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_SST(args.path, seed=args.seed)
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

    model = Model(args, emb_layer, nclasses).cuda()
    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr = args.lr
    )

    best_valid = 1e+8
    test_err = 1e+8
    tag = "model_cnn_cv_{}_dropout_{}_seed_{}_lr_{}".format(
        args.cv, args.dropout, args.seed, args.lr)
    pred_file = os.path.join("{tag}.pred".format(tag=tag))
    prob_file = os.path.join("{tag}.prob".format(tag=tag))

    # Normal training
    if not args.snapshot:
        for epoch in range(args.max_epoch):
            best_valid, test_err = train_model(epoch, model, optimizer,
                train_x, train_y,
                valid_x, valid_y,
                test_x, test_y,
                best_valid, test_err,
                pred_file
            )
            if args.lr_decay>0:
                optimizer.param_groups[0]['lr'] *= args.lr_decay

    # Snapshot ensembling
    else:
        # Modified from https://github.com/moskomule/pytorch.snapshot.ensembles
        logger.info("Using snapshot ensembling.")
        snapshot_dir = os.path.join(args.out, "snapshots")
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        cycles = args.cycles
        epochs = args.max_epoch
        epochs_per_cycle = epochs // cycles
        if epochs % cycles != 0:
            logger.warning("Total number of epochs is not divisible by number of cycles.")
        epoch = 0
        for cycle in range(cycles):
            logger.info("Cycle {cycle}".format(cycle=cycle+1))
            cycle_tag = "{tag}_cycle_{cycle}".format(tag=tag, cycle=cycle)
            for cycle_epoch in range(epochs_per_cycle):
                lr = cyclic_lr(args.lr, cycle_epoch, epochs_per_cycle)
                # Update learning rate to cyclic learning rate
                optimizer.param_groups[0]['lr'] = lr
                best_valid, test_err = train_model(epoch, model, optimizer,
                    train_x, train_y,
                    valid_x, valid_y,
                    test_x, test_y,
                    best_valid, test_err,
                    pred_file, prob_file
                )
                # Running count of total epochs
                epoch += 1

            # Save preds and probs for each snapshot
            pred_file = os.path.join(snapshot_dir, "{cycle_tag}.pred".format(cycle_tag=cycle_tag))
            prob_file = os.path.join(snapshot_dir, "{cycle_tag}.prob".format(cycle_tag=cycle_tag))
            eval_model(model, test_x, test_y, pred_file, prob_file)

            # Checkpoint model at the end of the cycle
            ckpt_file = os.path.join(snapshot_dir, "{cycle_tag}.ckpt".format(cycle_tag=cycle_tag))
            save(model, optimizer, epoch, ckpt_file)


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
    argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")
    argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
    argparser.add_argument("--la", action='store_true', help="whether to use la")
    argparser.add_argument("--dataset", type=str, default="mr", help="which dataset")
    argparser.add_argument("--path", type=str, required=True, help="path to corpus directory")
    argparser.add_argument("--embedding", type=str, required=True, help="word vectors")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--d", type=int, default=128)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--depth", type=int, default=2)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0.0)
    argparser.add_argument("--cv", type=int, default=0)
    argparser.add_argument("--seed", type=int, default=1234)
    argparser.add_argument("--out", type=str, help="Path to output directory.")
    argparser.add_argument("--snapshot", action='store_true', help="Use snapshot ensembling")
    argparser.add_argument("--cycles", type=int, help="Number of cycles/snapshots to take")
    args = argparser.parse_args()

    # Set random seed for torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Set random seed for numpy
    np.random.seed(seed=int(args.seed))

    # Dump command line arguments
    logger.info("Machine: " + os.uname()[1])
    logger.info("CMD: python " +  " ".join(sys.argv))
    print_key_pairs(args.__dict__.items(), title="Command Line Args", print_function=logger.info)
    # print (args)
    main(args)
