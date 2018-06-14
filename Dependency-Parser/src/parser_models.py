from rnn_builder import DeepBiLSTMBuilder, orthonormal_VanillaLSTMBuilder
from models.seq_head_sel_decoder import SeqHeadSelDecoder
from models.token_representation import TokenRepresentation
from models.biaffine_decoder import BiaffineDecoder
import pickle
import _dynet as dy

class HeadSelectorParser(object):

    def __init__(self, cfg, statistic, pc, load_model=False):
        if not load_model:
            self.get_tokenvec = TokenRepresentation(
                pc, 
                statistic['N_WORD'], cfg.WORD_DIM,
                statistic['N_UPOS'], cfg.UPOS_DIM,
                statistic['N_XPOS'], cfg.XPOS_DIM,
                cfg.EMB_FILE, statistic['word'],
                statistic['N_CHAR'], cfg.CHAR_DIM,
                cfg.N_FILTER, cfg.WIN_SIZES)

            self.encoder = DeepBiLSTMBuilder(
                pc,
                cfg.ENC_LAYERS,
                self.get_tokenvec.token_dim,
                cfg.ENC_H_DIM,
                dy.VanillaLSTMBuilder,
                param_init=True)

            self.decoder = SeqHeadSelDecoder(
                pc, 
                cfg.DEC_LAYERS, cfg.DEC_X_DIM, cfg.DEC_H_DIM,
                statistic['N_REL'], cfg.REL_DIM,
                dy.VanillaLSTMBuilder)
            
            self.cfg = cfg
            self.pc = pc
            self.statistic = statistic
        else:
            self.load()

    def run(self, sents, train=False):
        cfg = self.cfg
        token_vecs = self.get_tokenvec(
            sents, 
            self.statistic['word_cnt'], self.statistic['w2i'], 
            self.statistic['u2i'], self.statistic['x2i'],
            self.statistic['c2i'], cfg.MAXN_CHAR, cfg.CNN_ACT, 
            train, auto_batch=True)
        last, hs = self.encoder(token_vecs)
        sent_loss = self.decoder(
            sents, last, hs, 
            self.statistic['r2i'], 
            self.statistic['i2r'], train)
        return sent_loss

    def load(self, cfg):
        self.cfg = cfg
        self.get_tokenvec, self.encoder, self.decoder = dy.load(cfg.MODEL_FILE, self.pc)
        with open(cfg.STATISTIC_FILE, 'r') as fp:
            self.statistic = pickle.load(fp)

    def save(self, MODEL_FILE, STATISTIC_FILE):
        dy.save(MODEL_FILE, [self.get_tokenvec, self.encoder, self.decoder])
        with open(STATISTIC_FILE, 'wb') as fp:
            pickle.dump(self.statistic, fp)


class BiaffineParser(object):

    def __init__(self, cfg, statistic, pc, load_model=False):
        if not load_model:
            self.get_tokenvec = TokenRepresentation(
                pc,
                statistic['N_WORD'], cfg.WORD_DIM,
                statistic['N_UPOS'], cfg.UPOS_DIM,
                statistic['N_XPOS'], cfg.XPOS_DIM,
                cfg.EMB_FILE, statistic['word'],
                statistic['N_CHAR'], cfg.CHAR_DIM,
                cfg.N_FILTER, cfg.WIN_SIZES)
            self.encoder = DeepBiLSTMBuilder(
                pc, 
                cfg.ENC_LAYERS,
                self.get_tokenvec.token_dim,
                cfg.ENC_H_DIM,
                orthonormal_VanillaLSTMBuilder,
                param_init=True,
                fb_fusion=True)

            mlp_config = {
                'arc_size' : cfg.MLP_ARC_SIZE,
                'sizes'  : cfg.MLP_SIZES,
                'act'    : dy.rectify,
                'bias'   : True,
                'dropout': cfg.MLP_DROPOUT
            }
            biaffine_config = {
                'arc': {
                    'h_bias'  : False,
                    's_bias'  : True,
                },
                'rel': {
                    'h_bias'  : True,
                    's_bias'  : True,
                }
            }            
            self.decoder = BiaffineDecoder(
                pc,
                mlp_config,
                len(statistic["i2r"]),
                biaffine_config)
            self.cfg = cfg
            self.pc = pc
            self.statistic = statistic
        else:
            self.load()

    def run(self, sents, train=False):
        cfg = self.cfg
        token_vecs = self.get_tokenvec(
            sents, 
            self.statistic['word_cnt'], self.statistic['w2i'], 
            self.statistic['u2i'], self.statistic['x2i'],
            self.statistic['c2i'], cfg.MAXN_CHAR, cfg.CNN_ACT, 
            cfg.EMB_DROPOUT, train, auto_batch=True)
        hs = self.encoder(
            token_vecs, dropout_x=cfg.RNN_X_DROPOUT, 
            dropout_h=cfg.RNN_H_DROPOUT, train=train)
        sent_loss = self.decoder(
            sents, hs, 
            self.statistic['r2i'], 
            self.statistic['i2r'], train)
        return sent_loss

    def load(self, cfg):
        self.cfg = cfg
        self.get_tokenvec, self.encoder, self.decoder = dy.load(cfg.MODEL_FILE, self.pc)
        with open(cfg.STATISTIC_FILE, 'r') as fp:
            self.statistic = pickle.load(fp)

    def save(self, MODEL_FILE, STATISTIC_FILE):
        dy.save(MODEL_FILE, [self.get_tokenvec, self.encoder, self.decoder])
        with open(STATISTIC_FILE, 'wb') as fp:
            pickle.dump(self.statistic, fp)