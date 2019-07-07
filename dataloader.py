import gzip
import os
import sys
import re
import random

import numpy as np
import torch
import pickle
import logging

def pad(sequences, pad_token='<pad>', pad_left=True):
    ''' input sequences is a list of text sequence [[str]]
        pad each text sequence to the length of the longest
    '''
    max_len = max(5,max(len(seq) for seq in sequences))
    if pad_left:
        return [ [pad_token]*(max_len-len(seq)) + seq for seq in sequences ]
    return [ seq + [pad_token]*(max_len-len(seq)) for seq in sequences ]

def create_one_batch(x, y, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda(), torch.LongTensor(y).cuda()

def create_one_batch_feat_input(x, y):
    # make sure each sample in x is in the shape [seq length, n_dim]
    lengths = [sample.shape[0] for sample in x]
    max_length = max(lengths)
    n_sample = len(lengths)
    n_dim = x[0].shape[-1]
    batch_x = torch.zeros([max_length, n_sample, n_dim])
    for i, (sample, length) in enumerate(zip(x, lengths)):
        batch_x[:length, i, :].copy_(torch.from_numpy(sample))
    batch_y = torch.LongTensor(y)
    batch_len = torch.LongTensor(lengths)
    # we keep the batch on cpu so that for the largest dataset
    # the memory would not explode
    return batch_x.contiguous(), batch_y.contiguous(), batch_len.contiguous()

# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))
        logging.info("minibatches are length sorted")

    x = [ x[i] for i in lst ]
    y = [ y[i] for i in lst ]

    sum_len = 0.0
    batches_x = [ ]
    batches_y = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx, by = create_one_batch(x[i*size:(i+1)*size], y[i*size:(i+1)*size], map2id)
        sum_len += len(bx)
        batches_x.append(bx)
        batches_y.append(by)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]
        batches_y = [ batches_y[i] for i in perm ]

    logging.info("{} batches, avg len: {:.1f}".format(
        nbatch, sum_len/nbatch
    ))

    return batches_x, batches_y

def create_batches_feat_input(x, y, batch_size, sort=False):
    lst = range(len(x))
    # sort sequences based on their length; necessary for SST
    if sort:
        assert len(x[0].shape) == 2, "currently only support 2d feature input / sample"
        lst = sorted(lst, key=lambda i: x[i].shape[0])
        logging.info("minibatches are length sorted")

    x = [ x[i] for i in lst ]
    y = [ y[i] for i in lst ]

    sum_len = 0.0
    batches_x = [ ]
    batches_y = [ ]
    batches_len = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        ## sanity check batch creation using the following two print lines
        # print("test before ", x[(i+1)*size-1].shape, x[(i+1)*size-1], x[(i+1)*size-1].sum(-1))
        bx, by, blen = create_one_batch_feat_input(x[i*size:(i+1)*size], y[i*size:(i+1)*size])
        # print("test after ", bx[:, -1, :], bx[:, -1, :].shape, bx.shape, bx[:, -1, :].sum(-1))
        sum_len += bx.shape[0]
        batches_x.append(bx)
        batches_y.append(by)
        batches_len.append(blen)
    assert nbatch == len(batches_x)
    assert nbatch == len(batches_y)
    assert nbatch == len(batches_len)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]
        batches_y = [ batches_y[i] for i in perm ]
        batches_len = [ batches_len[i] for i in perm ]

    logging.info("{} batches, avg len: {:.1f}".format(
        nbatch, sum_len/nbatch
    ))

    return batches_x, batches_y, batches_len

def load_embedding_npz(path):
    data = np.load(path)
    return [ str(w) for w in data['words'] ], data['vals']

# TODO: This is largely copy-pasted from utils.py. One file should import from the other.
def load_embedding_txt(path, word_dict):
    """
    Loads a GloVe or FastText format embedding at specified path. Returns a
    vector of strings that represents the vocabulary and a 2-D numpy matrix that
    is the embeddings.
    """
    logging.info('Beginning to load embeddings')
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        wordlist = []
        embeddings = []
        if is_fasttext_format(lines): lines = lines[1:]
        for line in lines:
            row = line.strip('\n').split(' ')
            word = row.pop(0)
            if word_dict is not None and word not in word_dict:
                continue
            wordlist.append(word)
            embeddings.append([float(i) for i in row])
        embeddings = np.array(embeddings)
    assert len(wordlist) == embeddings.shape[0], 'Embedding dim must match wordlist length.'
    logging.info('Finished loading embeddings')
    return wordlist, embeddings

# TODO: This is copy-pasted from utils.py. One file should import from the other.
def is_fasttext_format(lines):
    first_line = lines[0].strip('\n').split(' ')
    return len(first_line) == 2 and first_line[0].isdigit() and first_line[1].isdigit()

def load_embedding_pkl(path):
    return pickle.load(open(path, "rb"))

def load_embedding(path, word_dict=None):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    elif path.endswith(".pkl"):
        return load_embedding_pkl(path)
    else:
        return load_embedding_txt(path, word_dict)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def split_and_save_dataset(dataset, input_data_dir, output_data_dir):
    train_x, train_y, valid_x, valid_y, test_x, test_y = split_dataset(dataset, input_data_dir)
    write_split_dataset(output_data_dir, train_x, train_y, valid_x, valid_y, test_x, test_y)

def split_dataset(dataset, data_dir):
    filenames = {
        'mr':'rt-polarity.all',
        'subj':'subj.all',
        'cr':'custrev.all',
        'sst':['stsa.binary.phrases.train','stsa.binary.dev','stsa.binary.test'],
        'sst1':['stsa.fine.phrases.train','stsa.fine.dev','stsa.fine.test'],
        'trec':['TREC.train.all','TREC.test.all'],
        'mpqa':'mpqa.all'
    }
    if dataset == 'sst' or dataset == 'sst1':
        dataset_paths = [os.path.join(data_dir, filenames[dataset][i]) for i in range(3)]
        train_x, train_y = read_dataset(dataset_paths[0], clean=False)
        valid_x, valid_y = read_dataset(dataset_paths[1], clean=False)
        test_x, test_y = read_dataset(dataset_paths[2], clean=False)
    else:
        if dataset == 'trec':
            dataset_paths = [os.path.join(data_dir, filenames[dataset][i]) for i in range(2)]
            train_valid_x, train_valid_y = read_dataset(dataset_paths[0], TREC=True)
            test_x, test_y = read_dataset(dataset_paths[1], TREC=True)
        else:
            assert dataset in ['mr','subj','cr','mpqa']
            dataset_path = os.path.join(data_dir, filenames[dataset])
            data, labels = read_dataset(dataset_path)
            train_valid_x, train_valid_y, test_x, test_y = random_split(data, labels)
        train_x, train_y, valid_x, valid_y = random_split(train_valid_x, train_valid_y)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def random_split(data, labels, frac_split=0.9):
    perm = list(range(len(labels)))
    random.shuffle(perm)
    M = int(len(labels) * frac_split)
    split1_x = [ data[i] for i in perm[:M] ]
    split1_y = [ labels[i] for i in perm[:M] ]
    split2_x = [ data[i] for i in perm[M:] ]
    split2_y = [ labels[i] for i in perm[M:] ]
    return split1_x, split1_y, split2_x, split2_y

def read_dataset(path, clean=True, TREC=False):
    data = []
    labels = []
    if sys.version_info[0] < 3:
        with open(path) as fin:
            for line in fin.readlines():
                label, sep, text = line.partition(' ')
                label = int(label)
                text = clean_str(text.strip(), TREC=TREC) if clean else text.strip()
                labels.append(label)
                data.append(text.split())
    else:
        with open(path, "r", encoding="ISO-8859-1") as fin:
            for line in fin.readlines():
                label, sep, text = line.partition(' ')
                label = int(label)
                text = clean_str(text.strip(), TREC=TREC) if clean else text.strip()
                labels.append(label)
                data.append(text.split())
    return data, labels

def write_dataset(path, data, labels):
    assert len(data) == len(labels)
    with open(path, 'w', encoding='ISO-8859-1') as f:
        for i in range(len(data)):
            f.write('{} {}\n'.format(labels[i], ' '.join(data[i])))

def read_split_dataset(data_dir, dataset):
    data_split_strs = ['train','heldout','test']
    data_list = [0]*3
    label_list = [0]*3
    for i in range(len(data_split_strs)):
        filename = '{}.{}.txt'.format(dataset, data_split_strs[i])
        dataset_path = os.path.join(data_dir, filename)
        # no need to clean, because we already did this before writing the files.
        data_list[i], label_list[i] = read_dataset(dataset_path, clean=False)
    return data_list[0], label_list[0], data_list[1], label_list[1],data_list[2], label_list[2]

def read_npy_dataset(data_dir, dataset):
    train_x = np.load("{}/{}.{}".format(data_dir, dataset, "train.feature.npz"))
    train_x = [train_x[name] for name in train_x.files]
    train_y = np.load("{}/{}.{}".format(data_dir, dataset, "train.label.npy"))
    valid_x = np.load("{}/{}.{}".format(data_dir, dataset, "heldout.feature.npz"))
    valid_x = [valid_x[name] for name in valid_x.files]    
    valid_y = np.load("{}/{}.{}".format(data_dir, dataset, "heldout.label.npy"))
    test_x = np.load("{}/{}.{}".format(data_dir, dataset, "test.feature.npz"))
    test_x = [test_x[name] for name in test_x.files]    
    test_y = np.load("{}/{}.{}".format(data_dir, dataset, "test.label.npy"))
    logging.info(" data length, {}, {}, {}".format(len(train_x), len(valid_x), len(test_x)))
    assert len(train_x) == len(train_y)
    assert len(valid_x) == len(valid_y)
    assert len(test_x) == len(test_y)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def write_split_dataset(data_dir, train_x, train_y, valid_x, valid_y, test_x, test_y):
    data_split_strs = ['train','heldout','test']
    data = [train_x, valid_x, test_x]
    labels = [train_y, valid_y, test_y]
    for i in range(len(data_split_strs)):
        filename = '{}.{}.txt'.format(dataset, data_split_strs[i])
        output_dataset_path = os.path.join(data_dir, filename)
        write_dataset(output_dataset_path, data[i], labels[i])

if __name__ == '__main__':
    random.seed(1)
    # datasets = ['mr','subj','cr','sst','trec','mpqa']
    datasets = ['sst1']
    input_data_dir = '/Users/Jian/Data/research/smallfry/src/smallfry/third_party/sent-conv-torch/data'
    output_data_dir = '/Users/Jian/Data/research/smallfry/src/smallfry/third_party/sentence_classification/data'
    # input_data_dir = 'C:\\Users\\avnermay\\git\\smallfry\\src\\third_party\\sent-conv-torch\\data'
    # output_data_dir = 'C:\\Users\\avnermay\\git\\smallfry\\src\\third_party\\sentence_classification\\data'
    for dataset in datasets:
        split_and_save_dataset(dataset, input_data_dir, output_data_dir)


# def read_MR(path, seed=1234):
#     file_path = os.path.join(path, "rt-polarity.all")
#     data, labels = read_dataset(file_path)
#     random.seed(seed)
#     perm = list(range(len(data)))
#     random.shuffle(perm)
#     data = [ data[i] for i in perm ]
#     labels = [ labels[i] for i in perm ]
#     return data, labels

# def read_SUBJ(path, seed=1234):
#     file_path = os.path.join(path, "subj.all")
#     data, labels = read_dataset(file_path)
#     random.seed(seed)
#     perm = list(range(len(data)))
#     random.shuffle(perm)
#     data = [ data[i] for i in perm ]
#     labels = [ labels[i] for i in perm ]
#     return data, labels

# def read_CR(path, seed=1234):
#     file_path = os.path.join(path, "custrev.all")
#     data, labels = read_dataset(file_path)
#     random.seed(seed)
#     perm = list(range(len(data)))
#     random.shuffle(perm)
#     data = [ data[i] for i in perm ]
#     labels = [ labels[i] for i in perm ]
#     return data, labels

# def read_MPQA(path, seed=1234):
#     file_path = os.path.join(path, "mpqa.all")
#     data, labels = read_dataset(file_path)
#     random.seed(seed)
#     perm = list(range(len(data)))
#     random.shuffle(perm)
#     data = [ data[i] for i in perm ]
#     labels = [ labels[i] for i in perm ]
#     return data, labels

# def read_TREC(path, seed=1234):
#     train_path = os.path.join(path, "TREC.train.all")
#     test_path = os.path.join(path, "TREC.test.all")
#     train_x, train_y = read_dataset(train_path, TREC=True)
#     test_x, test_y = read_dataset(test_path, TREC=True)
#     random.seed(seed)
#     perm = list(range(len(train_x)))
#     random.shuffle(perm)
#     train_x = [ train_x[i] for i in perm ]
#     train_y = [ train_y[i] for i in perm ]
#     return train_x, train_y, test_x, test_y

# def read_SST(path, seed=1234):
#     train_path = os.path.join(path, "stsa.binary.phrases.train")
#     valid_path = os.path.join(path, "stsa.binary.dev")
#     test_path = os.path.join(path, "stsa.binary.test")
#     train_x, train_y = read_dataset(train_path, False)
#     valid_x, valid_y = read_dataset(valid_path, False)
#     test_x, test_y = read_dataset(test_path, False)
#     random.seed(seed)
#     perm = list(range(len(train_x)))
#     random.shuffle(perm)
#     train_x = [ train_x[i] for i in perm ]
#     train_y = [ train_y[i] for i in perm ]
#     return train_x, train_y, valid_x, valid_y, test_x, test_y

# def cv_split(data, labels, nfold, test_id):
#     assert (nfold > 1) and (test_id >= 0) and (test_id < nfold)
#     lst_x = [ x for i, x in enumerate(data) if i%nfold != test_id ]
#     lst_y = [ y for i, y in enumerate(labels) if i%nfold != test_id ]
#     test_x = [ x for i, x in enumerate(data) if i%nfold == test_id ]
#     test_y = [ y for i, y in enumerate(labels) if i%nfold == test_id ]
#     perm = list(range(len(lst_x)))
#     random.shuffle(perm)
#     M = int(len(lst_x)*0.9)
#     train_x = [ lst_x[i] for i in perm[:M] ]
#     train_y = [ lst_y[i] for i in perm[:M] ]
#     valid_x = [ lst_x[i] for i in perm[M:] ]
#     valid_y = [ lst_y[i] for i in perm[M:] ]
#     return train_x, train_y, valid_x, valid_y, test_x, test_y

# def cv_split2(data, labels, nfold, valid_id):
#     assert (nfold > 1) and (valid_id >= 0) and (valid_id < nfold)
#     train_x = [ x for i, x in enumerate(data) if i%nfold != valid_id ]
#     train_y = [ y for i, y in enumerate(labels) if i%nfold != valid_id ]
#     valid_x = [ x for i, x in enumerate(data) if i%nfold == valid_id ]
#     valid_y = [ y for i, y in enumerate(labels) if i%nfold == valid_id ]
#     return train_x, train_y, valid_x, valid_y
