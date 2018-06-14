from collections import Counter
import cPickle as pickle
import re
import numpy as np
from k_means import KMeans

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");

def normalize(word):
    return word.lower()
    # return 'NUM' if numberRegex.match(word) else word.lower()


class Token(object):
    
    def __init__(self, id, form, lemma, upos, xpos, head=None, deprel=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.upos = upos.upper()
        self.xpos = xpos.upper()
        self.head = head
        self.deprel = deprel

        self.pred_head = None
        self.pred_deprel = None

        self.chars = None
        self.sons = None

    def __str__(self):
        values = [str(self.id), self.form, None, self.upos, self.xpos, None, 
                  str(self.pred_head) if self.pred_head is not None else None,
                  self.pred_deprel if self.pred_deprel is not None else None,
                  None, None]
        return '\t'.join(['_' if v is None else v for v in values])


class ParseForest(object):

    def __init__(self, sentence):
        self.roots = list(sentence)

        for root in self.roots:
            root.children = []
            root.scores = None
            root.vecs = None
            root.lstms = None

    def __len__(self):
        return len(self.roots)

    def attach(self, head_index, child_index):
        head = self.roots[head_index]
        child = self.roots[child_index]

        child.pred_head = head.id
        del self.roots[child_index]


def isProj(sentence):
    forest = ParseForest(sentence)
    unassigned = {
        entry.id: sum([1 for pentry in sentence 
                         if pentry.head == entry.id]) 
                  for entry in sentence
    }

    for _ in xrange(len(sentence)):
        for i in xrange(len(forest.roots) - 1):
            if (forest.roots[i].head == forest.roots[i+1].id 
                and unassigned[forest.roots[i].id] == 0):
                unassigned[forest.roots[i+1].id] -= 1
                forest.attach(i+1, i)
                break
            if (forest.roots[i+1].head == forest.roots[i].id 
                and unassigned[forest.roots[i+1].id] == 0):
                unassigned[forest.roots[i].id] -= 1
                forest.attach(i, i+1)
                break
        if len(forest.roots) == 1:
            return True
    return False


def build_dataset(conllu_path, maxn_char, nonproj=False, train=False):

    if train:
        wordsCount = Counter()
        uposCount = Counter()
        xposCount = Counter()
        relCount = Counter()
        charCount = Counter()
        charCount.update(['<SOW>', '<EOW>', '<PAD>', '<MUL>'])

    dataset = []

    with open(conllu_path, 'r') as conlluFP:
        for sentence in read_conll(conlluFP):
            sent = []
            for node in sentence:
                if isinstance(node, Token):
                    sent.append(node)
                    
                    if train:
                        wordsCount.update([node.norm])
                        uposCount.update([node.upos])
                        xposCount.update([node.xpos])
                        charCount.update(node.norm)
                        if node.head >= 0:
                            relCount.update([node.deprel])
                    
                    l = len(node.norm)
                    n = maxn_char-2
                    if l <= 30:
                        node.chars = ['<PAD>'] * ((n-l)/2)
                        node.chars.append('<SOW>')
                        node.chars.extend(list(node.norm))
                        node.chars.append('<EOW>')
                        node.chars.extend(['<PAD>'] * ((n-l+1)/2))
                    else:
                        node.chars = ['<SOW>']
                        node.chars.extend(list(node.norm[0:n/2-1]))
                        node.chars.append('<MUL>')
                        node.chars.extend(list(node.norm[-(n/2):]))
                        node.chars.append('<EOW>')

            for node in sent:
                node.sons = []

            for idx, node in enumerate(sent):
                if node.head != -1:
                    sent[node.head].sons.append(idx)

            if not train or nonproj or isProj(sent):
                dataset.append(sent)

    if train:
        word_cnt = wordsCount
        word = wordsCount.keys()
        # 0 -> UNK
        w2i = {w: i+1 for i, w in enumerate(word)}
        c2i = {c: i+1 for i, c in enumerate(charCount.keys())}
        x2i = {x: i+1 for i, x in enumerate(xposCount.keys())}
        u2i = {u: i+1 for i, u in enumerate(uposCount.keys())}
        r2i = {r: i for i, r in enumerate(relCount.keys())}
        i2r = {i: r for i, r in enumerate(relCount.keys())}
        return (dataset, {'word_cnt': word_cnt, 'word': word, 'w2i': w2i, 'c2i': c2i, 'x2i': x2i, 'u2i': u2i, 'r2i': r2i, 'i2r': i2r})
    else:
        return dataset
    

def read_conll(fh):
    tokens = [Token(0, '*root*', '*root*', 'RT-UPOS', 'RT-XPOS', -1, 'rroot')]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1:
                yield tokens
                tokens = [Token(0, '*root*', '*root*', 'RT-UPOS', 'RT-XPOS', -1, 'rroot')]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0] or not tok[0].isdigit():
                tokens.append(line.strip())
            else:
                tokens.append(Token(int(tok[0]), tok[1], tok[2], tok[3],
                                         tok[4],
                                         int(tok[6]) if tok[6] != '_' else -1,
                                         tok[7]))
    if len(tokens) > 1:
        yield tokens


class DataLoader(object):
    def __init__(self, sents, n_bkts):
        
        len_counter = Counter()
        for sent in sents:
            len_counter[len(sent)] += 1 
        self._bucket_sizes = KMeans(n_bkts, len_counter).splits
        self._buckets = [[] for i in xrange(n_bkts)]
        len2bkt = {}
        prev_size = -1
        for bkt_idx, size in enumerate(self._bucket_sizes):
            len2bkt.update(zip(range(prev_size+1, size+1), [bkt_idx] * (size - prev_size)))
            prev_size = size

        self._record = []
        for sent in sents:
            bkt_idx = len2bkt[len(sent)]
            self._buckets[bkt_idx].append(sent)
            idx = len(self._buckets[bkt_idx]) - 1
            self._record.append((bkt_idx, idx))

    def get_batches(self, batch_size, shuffle=True):
        batches = []
        for bkt_idx, bucket in enumerate(self._buckets):
            bucket_len = len(bucket)
            print bucket_len, self._bucket_sizes[bkt_idx]
            n_tokens = bucket_len * self._bucket_sizes[bkt_idx]
            n_splits = max(n_tokens // batch_size, 1)
            range_func = np.random.permutation if shuffle else np.arange
            for bkt_batch in np.array_split(range_func(bucket_len), n_splits):
                batches.append((bkt_idx, bkt_batch))
        
        if shuffle:
            np.random.shuffle(batches)
        return batches

    def get_sents(self, batch):
        bkt_idx, sents = batch
        return [self._buckets[bkt_idx][sent_idx] for sent_idx in sents]

    def get_dataset(self):
        return [self._buckets[bkt_idx][sent_idx] for bkt_idx, sent_idx in self._record]


if __name__ == '__main__':
    TRAIN_FILE = DEV_FILE = '../data/dev_v3.3.0.conllu'
    MAXN_CHAR = 30
    trainset, statistic = build_dataset(TRAIN_FILE, MAXN_CHAR, nonproj=True, train=True)
    print len(trainset), trainset[0][0].norm
    dev_sents = build_dataset(DEV_FILE, MAXN_CHAR, train=False)
    dev_loader = DataLoader(dev_sents, 5)
    devset = dev_loader.get_batches(500)
    print devset[0]
    print dev_loader.get_sents(devset[0])[0][1]
    devset = dev_loader.get_dataset()
    print len(devset), len(devset[0])
    for token in devset[0]:
        print token
