import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel

def deep_iter(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for u in x:
            for v in deep_iter(u):
                yield v
    else:
        yield x

class CNN_Text(nn.Module):

    def __init__(self, n_in, widths=[3,4,5], filters=100):
        super(CNN_Text,self).__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        # x is (batch, len, d)
        x = x.unsqueeze(1) # (batch, Ci, len, d)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(batch, Co, len), ...]        
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]
        x = torch.cat(x, 1)
        return x

class BertEmbeddingLayer(nn.Module):
    def __init__(self, bert_model_name='bert-base-cased', tokenizer=None):
        super(BertEmbeddingLayer, self).__init__()
        self.tokenizer = (tokenizer if tokenizer else
                          BertTokenizer.from_pretrained(bert_model_name, do_lower_case='uncased' in bert_model_name))
        self.model = BertModel.from_pretrained(bert_model_name)
        self.model.eval()
        self.n_d = 768 # dimension of BERT contextual embeddings (output of last hidden layer)

    def forward(self, input_ids, input_masks):
        with torch.no_grad():
            # input_ids and input_masks are LongTensors of dimensions (# sentences) x (# tokens).
            # all_encoder layers is (# layers) x (# sentences) x (# tokens) x (embedding dim),
            # where the first dimension (layers) is a list. List[FloatTensor].
            all_encoder_layers, _ = self.model(input_ids, token_type_ids=None, attention_mask=input_masks)
            # get last hidden layer, and detach it from computation graph (no backprop into BERT embeddings).
            embeddings =  all_encoder_layers[-1].detach()
            # FOR DEBUGGING
            # print('BertEmbeddingLayer: len(all_encoder_layers) = {}'.format(len(all_encoder_layers)))
            # print('BertEmbeddingLayer: embeddings.shape = {}'.format(embeddings.shape))
        return embeddings

class EmbeddingLayer(nn.Module):
    def __init__(self, n_d, words, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True):
        super(EmbeddingLayer, self).__init__()
        word2id = {}
        if embs is not None:
            embwords, embvecs = embs
            for word in embwords:
                assert word not in word2id, "Duplicate words in pre-trained embeddings"
                word2id[word] = len(word2id)

            logging.info("{} pre-trained word embeddings loaded.".format(len(word2id)))
            if n_d != len(embvecs[0]):
                logging.warn("n_d ({}) != word vector size ({}). Use {} for embeddings.".format(
                    n_d, len(embvecs[0]), len(embvecs[0])
                ))
                n_d = len(embvecs[0])

        for w in deep_iter(words):
            if w not in word2id:
                word2id[w] = len(word2id)

        if oov not in word2id:
            word2id[oov] = len(word2id)

        if pad not in word2id:
            word2id[pad] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        logging.info("Number of vectors: {}, Number of loaded vectors: {}, Number of oov {}".format(
            self.n_V, len(embwords), self.n_V - len(embwords)))
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        if embs is not None:
            weight  = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            logging.info("embedding shape: {}".format(weight.size()))

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2,1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, input):
        return self.embedding(input)
