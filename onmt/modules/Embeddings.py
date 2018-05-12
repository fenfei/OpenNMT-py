import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

from onmt.modules import Elementwise
from onmt.Utils import aeq


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        # We must wrap the self.pe in Variable to compute, not the other
        # way - unwrap emb(i.e. emb.data). Otherwise the computation
        # wouldn't be watched to build the compute graph.
        emb = emb * math.sqrt(self.dim)
        emb = emb + Variable(self.pe[:emb.size(0)], requires_grad=False)
        emb = self.dropout(emb)
        return emb


class GumbelSenseAttention(nn.Module):
    def __init__(self, cuda=True):
        super(GumbelSenseAttention, self).__init__()
        self.cuda = cuda

    def forward(self, pivots, contexts, mask=None, tau=0.5, scale=0.5):
        # pivots has shape (batch, d_word, num_senses)
        # contexts has shape (batch, 2*window_size, d_word)
        prod = torch.bmm(torch.mean(contexts, dim=1, keepdim=True), pivots).squeeze()
        if self.cuda:
            U = Variable(torch.rand(prod.size()).cuda(), requires_grad=False)
        else:
            U = Variable(torch.rand(prod.size()), requires_grad=False)
        y = prod - scale*torch.log(-torch.log(U + 1e-20) + 1e-20) # logits + scaled gumbel noise
        if mask is not None:
            y.data.masked_fill_(mask, -float('inf'))
        att = F.softmax(y/tau, dim=1) # (batch, num_senses)
        return att


class FlatTensorEmbedding(nn.Embedding):
    def __init__(self, word_vocab_size, num_senses, d_word, u=0.5, padding_idx=None, sparse=False):
        self.num_senses = num_senses
        self.d_word = d_word
        self.u = u
        super(FlatTensorEmbedding, self).__init__(word_vocab_size, num_senses*d_word, padding_idx=padding_idx, sparse=sparse)

    def reset_parameters(self):
        self.weight.data.uniform_(-self.u/self.d_word, self.u/self.d_word)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class WeightedSenseEmbedding(nn.Module):
    def __init__(self, word_vocab_size, num_senses, d_word, u=0.5, padding_idx=None, cuda=True):
        super(WeightedSenseEmbedding, self).__init__()
        self.d_word = d_word
        self.num_senses = num_senses
        self.SenseEmb = FlatTensorEmbedding(word_vocab_size=word_vocab_size, num_senses=num_senses, d_word=d_word, u=u)
        self.ContextEmb = FlatTensorEmbedding(word_vocab_size, 1, d_word=d_word, u=u, padding_idx=padding_idx)
        self.SenseAttention = GumbelSenseAttention(cuda=cuda)

    def forward(self, pivots, contexts, neg=None, tau=0.5, scale=0.5):
        # pivots - LongTensor with size L*B
        # contexts - LongTensor with size L*B*2*window_size
        # contexts - LongTensor with size (L*B)*neg_size
        psize = pivots.size()
        sz = psize[0]*psize[1]
        pivots = self.SenseEmb(pivots) # L*B*(d_word*num_senses)
        pivots = pivots.view(sz, self.d_word, -1) # (L*B) * d_word * num_senses
        contexts = contexts.view(sz, -1) #(L*B)*(window_size*2)
        contexts = self.ContextEmb(contexts) #(L*B)*(window_size*2)*d_word
        att = self.SenseAttention(pivots, contexts, tau=tau, scale=scale).unsqueeze(2) # (L*B)*num_senses*1
        emb = torch.bmm(pivots, att).view(psize[0], psize[1], -1)
        if neg is None:
            return emb
        neg_multi = self.ContextEmb(neg) # (L*B)*neg_size*d_word
        att = att.squeeze().unsqueeze(1) # (L*B)*1*num_senses
        corr_scores_m = att*torch.log(1e-20+torch.sigmoid(torch.bmm(contexts, pivots)))
        neg_scores_m = att*torch.log(1e-20 + torch.sigmoid(-torch.bmm(neg_multi, pivots)))
        loss = -torch.sum(corr_scores_m) - torch.sum(neg_scores_m)
        return emb, loss


class Embeddings(nn.Module):
    """
    Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx (list of int): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        num_senses (int, optional, default -1): number of senses considered for
                                                each word, if -1 then use the normal
                                                word embedding layer.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.

        position_encoding (bool): see :obj:`onmt.modules.PositionalEncoding`

        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    `-feat_merge mlp`
        dropout (float): dropout probability.
    """
    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
		 num_senses=-1,
                 position_encoding=False,
                 feat_merge="concat",
                 feat_vec_exponent=0.7, feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 sparse=False):

        self.word_padding_idx = word_padding_idx

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = list(zip(vocab_sizes, emb_dims, pad_indices))
        if num_senses > 1:
            embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
                          for vocab, dim, pad in emb_params[1:]]
        else:
            embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
                          for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        if num_senses > 1:
            self.SenseModule = WeightedSenseEmbedding(word_vocab_size, num_senses, word_vec_size, padding_idx=word_padding_idx)
        else:
            self.SenseModule = None
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            out_dim = word_vec_size
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        if position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

    @property
    def word_lut(self):
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, input, contexts=None, neg=None, tau=0.5, scale=0.5):
        """
        Computes the embeddings for words and features.

        Args:
            input (`LongTensor`): index tensor `[len x batch x nfeat]`
        Return:
            `FloatTensor`: word embeddings `[len x batch x embedding_size]`
        """

        in_length, in_batch, nfeat = input.size()
        
        word_emb, sense_loss = None, None
        if self.SenseModule is not None:
            aeq(nfeat-1, len(self.emb_luts))
            assert contexts is not None
            if neg is None:
                word_emb = self.SenseModule(input[:,:,0], contexts, tau=tau, scale=scale)
            else:
                word_emb, sense_loss = self.SenseModule(input[:,:,0], contexts, neg=neg, tau=tau, scale=scale)
        else:
            aeq(nfeat, len(self.emb_luts))
        emb = self.make_embedding((input, word_emb))

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_batch, out_batch)
        aeq(emb_size, self.embedding_size)
        if neg is None:
            return emb
        else:
            return emb, sense_loss

