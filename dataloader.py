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

def get_token_ids_and_masks(tokenized_sentences, tokenizer):
    max_sentence_length = max(5,max(len(tokenized_sentence) for tokenized_sentence in tokenized_sentences))
    input_ids = []
    input_masks = []
    for sentence in tokenized_sentences:
        sentence_input_ids = tokenizer.convert_tokens_to_ids(sentence)
        sentence_input_mask = [1] * len(sentence_input_ids)
        # Zero-pad up to the sequence length.
        while len(sentence_input_ids) < max_sentence_length:
            sentence_input_ids.append(0)
            sentence_input_mask.append(0)
        input_ids.append(sentence_input_ids)
        input_masks.append(sentence_input_mask)
    return input_ids, input_masks

def create_one_batch(x, y, map2id, oov='<oov>', tokenizer=None):
    if tokenizer:
        # This is true when use_bert_embeddings is true.
        batch_input_ids, batch_input_masks = get_token_ids_and_masks(x, tokenizer)
        batch_input_ids = torch.LongTensor(batch_input_ids).t().contiguous().cuda()
        batch_input_masks = torch.LongTensor(batch_input_masks).t().contiguous().cuda()
        y = torch.LongTensor(y).cuda()
        return (batch_input_ids, batch_input_masks), y
    else:
        oov_id = map2id[oov]
        x = pad(x)
        length = len(x[0])
        batch_size = len(x)
        x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
        x = torch.LongTensor(x)
        assert x.size(0) == length*batch_size
        return x.view(batch_size, length).t().contiguous().cuda(), torch.LongTensor(y).cuda()

def tokenize(sentences, dataset, tokenizer):
    tokenized_sentences = []
    for sentence in sentences:
        clean_sentence = clean_str(sentence, dataset)
        words = clean_sentence.split()
        if tokenizer:
            tokens = []
            tokens.append('[CLS]')
            for word in words:
                tokens.extend(tokenizer.tokenize(word))
            tokens.append('[SEP]')
            assert len(tokens) - 2 >= len(words)
        else:
            tokens = words
        tokenized_sentences.append(tokens)
    return tokenized_sentences

# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, map2id, perm=None, sort=False, tokenizer=None, write=False, out=""):
    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))
        
        logging.info("minibatches are length sorted")

    # lst index to orig index
    map_len = {}
    
    x = [ x[i] for i in lst ]
    y = [ y[i] for i in lst ]
    ind = [ i for i in lst ] 

    sum_len = 0.0
    batches_x = [ ]
    batches_y = [ ]
    order = {}
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx, by = create_one_batch(x[i*size:(i+1)*size], y[i*size:(i+1)*size], map2id, tokenizer=tokenizer)
        order[i] = ind[i*size:(i+1)*size]
        sum_len += len(by)
        batches_x.append(bx)
        batches_y.append(by)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]
        batches_y = [ batches_y[i] for i in perm ]
        
        if write:
            filename = out + "_raw"
            with open(filename, 'w') as f:
                for batch_i in perm:
                    for j in order[batch_i]:
                        f.write(str(j) + "\n")

    logging.info("{} batches, avg len: {:.1f}".format(
        nbatch, sum_len/nbatch
    ))

    return batches_x, batches_y

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

def clean_str(string, dataset, ignore_dataset=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = string.strip()
    if 'sst' in dataset and not ignore_dataset:
        return string
    else:
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
        return string.strip() if (dataset=='trec' and not ignore_dataset) else string.strip().lower()

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
        'mpqa':'mpqa.all',
       
        #augmented sets
        'mr_alpha,0.1':'rt-polarity.all',
        'subj_alpha,0.1':'subj.all',
        'cr_alpha,0.1':'custrev.all',
        'mpqa_alpha,0.1':'mpqa.all',
        'mr_alpha,0.2':'rt-polarity.all',
        'subj_alpha,0.2':'subj.all',
        'cr_alpha,0.2':'custrev.all',
        'mpqa_alpha,0,2':'mpqa.all',

        #gpt2 datasets
        'transformerlm_gentext_dataset,cr_4x': 'custrev.all',
        'transformerlm_gentext_dataset,mpqa_4x': 'mpqa.all',
        'transformerlm_gentext_dataset,mr_4x': 'rt-polarity.all',
        'transformerlm_gentext_dataset,sst_4x': ['stsa.binary.phrases.train','stsa.binary.dev','stsa.binary.test'],
        'transformerlm_gentext_dataset,subj_4x': 'subj.all',
        'transformerlm_gentext_dataset,trec_4x': ['TREC.train.all','TREC.test.all']
    }

    if dataset == 'sst' or dataset == 'sst1':
        dataset_paths = [os.path.join(data_dir, filenames[dataset][i]) for i in range(3)]
        train_x, train_y = read_dataset(dataset_paths[0], dataset)
        valid_x, valid_y = read_dataset(dataset_paths[1], dataset)
        test_x, test_y = read_dataset(dataset_paths[2], dataset)
    else:
        if dataset == 'trec':
            dataset_paths = [os.path.join(data_dir, filenames[dataset][i]) for i in range(2)]
            train_valid_x, train_valid_y = read_dataset(dataset_paths[0], dataset)
            test_x, test_y = read_dataset(dataset_paths[1], dataset)
        else:
            original_datasets = ['mr','subj','cr','mpqa']
            augmented_datasets = ['mr_alpha,0.1', 'mr_alpha,0.2', 'cr_alpha,0.1', 'cr_alpha,0.2', 'subj_alpha,0.1', 'subj_alpha,0.2', 'mpqa_alpha,0.1', 'mpqa_alpha,0.2']
            gpt2_datasets = ['transformerlm_gentext_dataset,cr_4x', 'transformerlm_gentext_dataset,mpqa_4x', 'transformerlm_gentext_dataset,mr_4x', 'transformerlm_gentext_dataset,sst_4x', 'transformerlm_gentext_dataset,subj_4x', 'transformerlm_gentext_dataset,trec_4x']
            assert dataset in original_datasets + augmented_datasets + gpt2_datasets
            dataset_path = os.path.join(data_dir, filenames[dataset])
            data, labels = read_dataset(dataset_path, dataset)
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

def read_dataset(path, dataset, clean=False, split=False):
    data = []
    labels = []
    if sys.version_info[0] < 3:
        with open(path) as fin:
            for line in fin.readlines():
                label, sep, text = line.partition(' ')
                label = int(label)
                text = clean_str(text, dataset) if clean else text.strip()
                labels.append(label)
                if split:
                    data.append(text.split())
                else:
                    data.append(text)
    else:
        with open(path, "r", encoding="ISO-8859-1") as fin:
            for line in fin.readlines():
                label, sep, text = line.partition(' ')
                label = int(label)
                text = clean_str(text, dataset) if clean else text.strip()
                labels.append(label)
                if split:
                    data.append(text.split())
                else:
                    data.append(text)
    return data, labels

def write_dataset(path, data, labels):
    assert len(data) == len(labels)
    with open(path, 'w', encoding='ISO-8859-1') as f:
        for i in range(len(data)):
            f.write('{} {}\n'.format(labels[i], data[i]))

def read_split_dataset(data_dir, dataset, trainfraction=1.0):
    data_split_strs = ['train','heldout','test']
    data_list = [0]*3
    label_list = [0]*3
    base = ""
    original_datasets = ['mr', 'cr', 'mpqa', 'subj', 'sst', 'trec']
    for dset in original_datasets:
        if dset in dataset:
            base = dset
    assert base != ""
    for i in range(len(data_split_strs)):
        if i > 0:
            filename = '{}.{}.txt'.format(base, data_split_strs[i])
            print("FILENAME: " + str(filename))
        else:
            filename = '{}.{}.txt'.format(dataset, data_split_strs[i])
        dataset_path = os.path.join(data_dir, filename)
        data_list[i], label_list[i] = read_dataset(dataset_path, dataset)
    
    x_train = []
    y_train = []
    if trainfraction < 1.0:
        x_train, y_train = downsample(data_list[0], label_list[0], trainfraction)
    else:
        x_train = data_list[0]
        y_train = label_list[0]
    return x_train, y_train, data_list[1], label_list[1], data_list[2], label_list[2]

def sample(length, trainfraction):
    import random
    sample_size = round(length * trainfraction)
    sample = random.sample(range(1, length), sample_size)
    return sample

def downsample(x_list, y_list, trainfraction):
    import math
    assert trainfraction < 1 and trainfraction > 0, 'Should only call downsample with trainfraction between 0 and 1'
    total_train_examples = len(x_list)
    random_indices = sample(total_train_examples, trainfraction)
    x_list_downsampled = [
        x_list[i]
        for i in random_indices
    ]
    y_list_downsampled = [
        y_list[i]
        for i in random_indices
    ]
    # Use multiple copies of downsampled data to keep total amount of training data the same.
    # This code support trainfractions such that 1/trainfraction is not an integer.
    int_multiple = math.ceil(1.0/trainfraction)
    x_list_downsampled = (x_list_downsampled * int_multiple)[:total_train_examples]
    y_list_downsampled = (y_list_downsampled * int_multiple)[:total_train_examples]
    return x_list_downsampled, y_list_downsampled

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
    datasets = ['mr','subj','cr','sst','trec','mpqa', 'sst1']
    # datasets = ['sst1']
    # input_data_dir = '/Users/Jian/Data/research/smallfry/src/smallfry/third_party/sent-conv-torch/data'
    # output_data_dir = '/Users/Jian/Data/research/smallfry/src/smallfry/third_party/sentence_classification/data'
    input_data_dir = 'C:\\Users\\Avner\\git\\smallfry_internal\\src\\smallfry\\third_party\\sent-conv-torch\\data'
    output_data_dir = 'C:\\Users\\Avner\\git\\smallfry_internal\\src\\smallfry\\third_party\\sentence_classification\\data'
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
