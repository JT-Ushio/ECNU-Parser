import os
import _dynet as dy
import numpy as np
import random
from char2word_embedder import Char2WordCNNEmbedder


# TODO(TaoJi): Add Pos tag dropout
class TokenRepresentation(object):
    """This builds token representation:

    :param model dynet.ParameterCollection: 
    :param n_word int: Number of word 
    :param word_dim int: Dimension of word embedding
    :param n_upos int: Number of upos
    :param upos_dim int: Dimension of upos embedding
    :param n_xpos int: Number of xpos 
    :param xpos_dim int: Dimension of xpos embedding
    :param word list: Word list
    :param n_char int: Number of char 
    :param char_dim int: Dimension of char embedding
    :param n_filter int: Number of CNN filter
    :param win_sizes list: Filter width list
    :returns: token_vecs
    :rtype: list
    """
    def __init__(self, model, n_word, word_dim, n_upos, upos_dim, n_xpos, xpos_dim, 
                 ex_emb_file, word, n_char, char_dim, n_filter, win_sizes):
        pc = model.add_subcollection()
        if word_dim:
            self.wlookup = pc.lookup_parameters_from_numpy(
                np.random.randn(n_word, word_dim).astype(np.float32))
        if upos_dim:
            self.uplookup = pc.lookup_parameters_from_numpy(
                np.random.randn(n_upos, upos_dim).astype(np.float32))
        if xpos_dim:
            self.xplookup = pc.lookup_parameters_from_numpy(
                np.random.randn(n_xpos, xpos_dim).astype(np.float32))

        # build pre-trained word Embedding
        if os.path.isfile(ex_emb_file):
            pword = {}
            with open(ex_emb_file, 'r') as fp:
                for w in fp:
                    w_list = w.strip().split(' ')
                    # if w_list[0] not in word: continue        # get all pre-trained word Embedding
                    pword[w_list[0]] =  [float(f) for f in w_list[1:]]
            pword_dim = len(pword.values()[0])
            miss_vec = [0.0 for _ in xrange(pword_dim)]
            self.extrnd = {w: i + 1 for i, w in enumerate(pword)}
            n_pword = len(self.extrnd) + 1
            self.pwlookup = pc.add_lookup_parameters((n_pword, pword_dim))
            self.pwlookup.init_row(0, miss_vec)
            for w, i in self.extrnd.iteritems():
                self.pwlookup.init_row(i, pword[w])
                    
            print 'Load pre-trained word embedding. Vector dims %d, Word nums %d' % (pword_dim, n_pword)
        else:
            pword_dim = 0

        self.token_dim = word_dim + upos_dim + xpos_dim + pword_dim
        if char_dim:
            self.c2w_emb = Char2WordCNNEmbedder(pc, n_char, char_dim, n_filter, win_sizes)
            self.clookup = pc.add_lookup_parameters((n_char, char_dim))
            self.token_dim += n_filter * len(win_sizes)

        self.word_dim = word_dim
        self.upos_dim = upos_dim
        self.xpos_dim = xpos_dim
        self.char_dim = char_dim
        self.pword_dim = pword_dim
        self.pc = pc
        self.spec = (n_word, word_dim, n_upos, upos_dim, n_xpos, xpos_dim, 
                     ex_emb_file, word, n_char, char_dim, n_filter, win_sizes)

    def __call__(self, sentences, word_cnt, w2i, u2i, x2i, c2i, maxn_char, 
                 act, emb_dropout, train=False, auto_batch=False):

        if auto_batch:
            sentence = sentences
            if self.char_dim:
                c2w_vec_list = self.c2w_emb(sentence, c2i, maxn_char, act, train)

            token_vecs = []
            for idx, token in enumerate(sentence):
                vec = []
                # get word Embedding (use word dropout)
                if self.word_dim:   
                    # wordc = float(word_cnt.get(token.norm, 0))
                    # word_dropout = not train or (random.random() < (wordc/(0.25+wordc)))   
                    word_dropout = not train or random.random() > emb_dropout
                    word_id = int(w2i.get(token.norm, 0))         
                    word_vec = self.wlookup[word_id if word_dropout else 0]
                    vec.append(word_vec)
                
                if self.pword_dim:
                    pword_id = int(self.extrnd.get(token.norm, 0))                    
                    pword_vec = self.pwlookup[pword_id if word_dropout else 0]
                    vec.append(pword_vec)

                if self.upos_dim:
                    upos_id = int(u2i.get(token.upos, 0))
                    upos_dropout = not train or random.random() > emb_dropout
                    upos_vec = self.uplookup[upos_id if upos_dropout else 0]
                    vec.append(upos_vec)

                if self.xpos_dim:
                    xpos_id = int(x2i.get(token.xpos, 0))
                    xpos_dropout = not train or random.random() > emb_dropout
                    xpos_vec = self.xplookup[xpos_id if xpos_dropout else 0]
                    vec.append(xpos_vec)

                if self.char_dim:
                    vec.append(c2w_vec_list[idx])

                token_vecs.append(dy.concatenate(vec))
            return token_vecs

        else:
            wids = []
            xids = []
            uids = []
            masks = []

            maxL = 0
            for sent in sentences:
                maxL = max(maxL, len(sent))

            for i in range(maxL):
                if self.word_dim:
                    wids.append([(w2i.get(sent[i].norm, 0) if len(sent) > i else w2i['<PAD>']) for sent in sentences])
                if self.xpos_dim:
                    xids.append([(x2i.get(sent[i].xpos, 0) if len(sent) > i else x2i['<PAD>']) for sent in sentences])
                if self.upos_dim:
                    uids.append([(u2i.get(sent[i].upos, 0) if len(sent) > i else u2i['<PAD>']) for sent in sentences])
                mask = [(1 if len(sent)>i else 0) for sent in sentences]
                masks.append(mask)

            tokens_batch = []
            # TODO: add Char Embedding Batch
            for i in range(maxL):
                vec = []
                if self.word_dim:
                    wids_batch = wids[i]
                    wvec = dy.lookup_batch(self.wlookup, wids_batch)
                    vec.append(wvec)
                if self.xpos_dim:
                    xids_batch = xids[i]
                    xvec = dy.lookup_batch(self.xplookup, xids_batch)
                    vec.append(xvec)
                if self.upos_dim:
                    uids_batch = uids[i]
                    uvec = dy.lookup_batch(self.uplookup, uids_batch)
                    vec.append(uvec)
                tokens_batch.append(dy.concatenate(vec))
            return maxL, masks, tokens_batch


    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.
        
        It is one of the prerequisites for Dynet save/load method.
        """
        (n_word, word_dim, n_upos, upos_dim, n_xpos, xpos_dim, 
         ex_emb_file, word, n_char, char_dim, n_filter, win_sizes) = spec
        return TokenRepresentation(
            model, n_word, word_dim, n_upos, upos_dim, n_xpos, xpos_dim, 
            ex_emb_file, word, n_char, char_dim, n_filter, win_sizes)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.
        
        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc