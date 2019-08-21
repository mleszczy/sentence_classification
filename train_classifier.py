import os
import sys
import argparse
import time
import pickle
import random
import math
from subprocess import check_output
import hashlib

import numpy as np
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# import cuda_functional as MF
sys.path.append(os.path.dirname(__file__))
import dataloader
import modules
from sentutils import *
sys.path.remove(os.path.dirname(__file__))
import logging

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
        if torch.__version__ >= '0.4':
            total_loss += loss.data*x.size(1)
        else:
            total_loss += loss.data[0]*x.size(1)
        pred = output.data.max(1)[1]
        correct += pred.eq(y.data).cpu().sum()
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
    if torch.__version__ >= '0.4':
        return 1.0-float(correct)/float(cnt)
    else:
        return 1.0-correct/cnt


def train_model(epoch, model, optimizer,
        train_x, train_y, valid_x, valid_y,
        test_x, test_y,
        best_valid, test_err,
        save_mdl=None,
        pred_file=None,
        prob_file=None):
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

    if torch.__version__ >= "0.4":
        logging.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_err={:.6f}".format(
            epoch, niter,
            optimizer.param_groups[0]['lr'],
            loss.data,
            valid_err
        ))
    else:
        logging.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_err={:.6f}".format(
            epoch, niter,
            optimizer.param_groups[0]['lr'],
            loss.data[0],
            valid_err
        ))

    if valid_err < best_valid:
        best_valid = valid_err
        test_err = eval_model(model, test_x, test_y, pred_file=pred_file, prob_file=prob_file)
        # Save model
        if save_mdl is not None:
            torch.save(model, save_mdl)
    return best_valid, test_err

def cyclic_lr(initial_lr, iteration, epoch_per_cycle):
    return initial_lr * (math.cos(math.pi * iteration / epoch_per_cycle) + 1) / 2

def main(args):
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_split_dataset(args.path, args.dataset)
    data = train_x + valid_x + test_x

    if args.embedding:
        logging.info("Using single embedding file.")
        emb_layer = modules.EmbeddingLayer(
            args.d, data,
            embs = dataloader.load_embedding(args.embedding, args.embed_type),
            normalize=not args.no_normalize
        )
    elif args.embedding_list:
        logging.info("Using embedding list.")
        embedding_list = []
        with open(args.embedding_list, 'r') as f:
            lines = f.readlines()
            assert len(lines) >= args.cycles
            for i, emb in enumerate(lines):
                logging.info("Embedding file: {embfile}".format(embfile=emb.strip()))
                embedding_list.append(modules.EmbeddingLayer(
                                    args.d, data,
                                    embs = dataloader.load_embedding(emb.strip()),
                                    normalize=not args.no_normalize).cuda())
        emb_layer = embedding_list[0]
        print("Embedding list length", len(embedding_list))
    else:
        raise ValueError("Need to provide embedding or list of embeddings.")

    orig_emb_layer = emb_layer

    nclasses = max(train_y)+1
    logging.info(str(nclasses) + " classes in total")

    train_x, train_y = dataloader.create_batches(
        train_x, train_y,
        args.batch_size,
        emb_layer.word2id,
        sort = 'sst' in args.dataset
        # sort = args.dataset == 'sst'
    )
    valid_x, valid_y = dataloader.create_batches(
        valid_x, valid_y,
        args.batch_size,
        emb_layer.word2id,
        sort = 'sst' in args.dataset
        # sort = args.dataset == 'sst'
    )
    test_x, test_y = dataloader.create_batches(
        test_x, test_y,
        args.batch_size,
        emb_layer.word2id,
        sort = 'sst' in args.dataset
        # sort = args.dataset == 'sst'
    )

    if args.load_mdl is None:
        model = Model(args, emb_layer, nclasses).cuda()
    else:
        # Note: this will overwrite all parameters
        model = torch.load(args.load_mdl).cuda()

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr = args.lr
    )

    best_valid = 1e+8
    test_err = 1e+8

    if not args.tag:
        if args.cnn:
            tag = "model_cnn_dropout_{dropout}_seed_{seed}_lr_{lr}".format(
                dropout=args.dropout, seed=args.model_seed, lr=args.lr)
        elif args.la:
             tag = "model_la_dropout_{dropout}_seed_{seed}_lr_{lr}".format(
                dropout=args.dropout, seed=args.model_seed, lr=args.lr)
        elif args.lstm:
            tag = "model_lstm_dropout_{dropout}_seed_{seed}_lr_{lr}".format(
                dropout=args.dropout, seed=args.model_seed, lr=args.lr)
    else: # enables deprecated use of naming
        tag = args.tag

    # pred_file = os.path.join(args.out, "{tag}.pred".format(tag=tag))
    # prob_file = os.path.join(args.out, "{tag}.prob".format(tag=tag))
    pred_file = None
    prob_file = None

    # Normal training
    if not args.snapshot:
        for epoch in range(args.max_epoch):
            best_valid, test_err = train_model(epoch, model, optimizer,
                train_x, train_y,
                valid_x, valid_y,
                test_x, test_y,
                best_valid, test_err,
                save_mdl=args.save_mdl,
                pred_file=pred_file,
                prob_file=prob_file
            )
            if args.lr_decay>0:
                optimizer.param_groups[0]['lr'] *= args.lr_decay

    # Snapshot ensembling
    else:
        # Modified from https://github.com/moskomule/pytorch.snapshot.ensembles
        logging.info("Using snapshot ensembling.")
        snapshot_dir = os.path.join(args.out, "snapshots_{}".format(args.cycles))

        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        cycles = args.cycles
        epochs = args.max_epoch
        epochs_per_cycle = epochs // cycles
        if epochs % cycles != 0:
            logging.warning("Total number of epochs is not divisible by number of cycles.")
        epoch = 0
        for cycle in range(cycles):
            logging.info("Cycle {cycle}".format(cycle=cycle))
            cycle_tag = "{tag}_cycle_{cycle}".format(tag=tag, cycle=cycle)
            for cycle_epoch in range(epochs_per_cycle):
                lr = cyclic_lr(args.lr, cycle_epoch, epochs_per_cycle)
                # Update learning rate to cyclic learning rate
                optimizer.param_groups[0]['lr'] = lr
                best_valid, test_err = train_model(epoch, model, optimizer,
                    train_x, train_y,
                    valid_x, valid_y,
                    test_x, test_y,
                    best_valid, test_err
                )
                # Running count of total epochs
                epoch += 1

            # Save preds and probs for each snapshot *at the end of the snapshot*
            pred_file = os.path.join(snapshot_dir, "{cycle_tag}.pred".format(cycle_tag=cycle_tag))
            prob_file = os.path.join(snapshot_dir, "{cycle_tag}.prob".format(cycle_tag=cycle_tag))
            eval_model(model, test_x, test_y, pred_file=pred_file, prob_file=prob_file)

            # Checkpoint model at the end of the cycle
            ckpt_file = os.path.join(snapshot_dir, "{cycle_tag}.ckpt".format(cycle_tag=cycle_tag))
            save(model, optimizer, epoch, ckpt_file)

            # Update embedding layer if there are multiple embeddings
            if args.embedding_list is not None:
                emb_layer = embedding_list[cycle]
                assert(emb_layer.word2id == orig_emb_layer.word2id)
                assert(emb_layer.n_d == orig_emb_layer.n_d)
                model.emb_layer = emb_layer

    logging.info("=" * 40)
    logging.info("best_valid: {:.6f}".format(
        best_valid
    ))
    logging.info("test_err: {:.6f}".format(
        test_err
    ))
    logging.info("=" * 40)
    return best_valid, test_err

def train_sentiment(cmdline_args):
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")
    argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
    argparser.add_argument("--la", action='store_true', help="whether to use la")
    argparser.add_argument("--no_normalize", action='store_true', help="Do not normalize embeddings")
    argparser.add_argument("--dataset", type=str, default="mr", help="which dataset")
    argparser.add_argument("--path", type=str, required=True, help="path to corpus directory")
    argparser.add_argument("--embedding", type=str, help="word vectors")
    argparser.add_argument("--embed_type", type=str, choices=['orig','rand','char','wordnet'], default='orig', help="Type of word vectors to use") # Added by Avner 8/20/19
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--d", type=int, default=128)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--depth", type=int, default=2)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0.0)
    # argparser.add_argument("--cv", type=int, default=0)
    argparser.add_argument("--model_seed", type=int, default=1234)
    argparser.add_argument("--data_seed", type=int, default=1234)
    argparser.add_argument("--save_mdl", type=str, default=None, help="Save model to this file.")
    argparser.add_argument("--load_mdl", type=str, default=None, help="Load model from this file.")
    argparser.add_argument("--out", type=str, help="Path to output directory.")
    argparser.add_argument("--snapshot", action='store_true', help="Use snapshot ensembling")
    argparser.add_argument("--cycles", type=int, help="Number of cycles/snapshots to take")
    argparser.add_argument("--embedding_list", type=str, help="List of word vector files")
    argparser.add_argument("--tag", type=str, help="Tag for naming files")
    argparser.add_argument("--no_cudnn", action="store_true", help="Turn off cuDNN for deterministic CNN")
    # argparser.add_argument("--no_cv", action="store_true", help="Merge train and validation dataset.")
    print(cmdline_args)
    args = argparser.parse_args(cmdline_args)

    # Dump git hash
    # h = check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    # logging.info("Git hash: " + h)

    # Dump embedding hash
    if args.embedding:
        embedding_hash = hashlib.md5(open(args.embedding, 'rb').read()).hexdigest()
        logging.info("Embedding hash: " + embedding_hash)

    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
        logging.info("CuDNN Disabled!!!")

    # Set random seed for torch
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed(args.model_seed)

    # Set random seed for numpy
    np.random.seed(seed=args.model_seed)

    # Dump command line arguments
    logging.info("Machine: " + os.uname()[1])
    logging.info("CMD: python " +  " ".join(sys.argv))
    print_key_pairs(args.__dict__.items(), title="Command Line Args", print_function=logging.info)
    # print (args)
    return main(args)
