from ConfigParser import SafeConfigParser
import argparse, os
import _dynet as dy

class BaseConfigurator(object):
    def __init__(self, config):
        self._config = config
        if not os.path.isdir(self.SAVE_DIR):
            os.mkdir(self.SAVE_DIR)

    @property
    def EMB_FILE(self):
        return self._config.get('Data', 'pretrained_embeddings_file')

    @property
    def TRAIN_FILE(self):
        return self._config.get('Data', 'train_file')    

    @property
    def DEV_FILE(self):
        return self._config.get('Data', 'dev_file')

    @property
    def TEST_FILE(self):
        return self._config.get('Data', 'test_file')

    @property
    def SAVE_DIR(self):
        return self._config.get('Save', 'save_dir')

    @property
    def LOG_FILE(self):
        return self._config.get('Save', 'log_file')

    @property
    def EVAL_DEV(self):
        return self._config.get('Save', 'eval_dev')

    @property
    def PRED_DEV(self):
        return self._config.get('Save', 'pred_dev')

    @property
    def EVAL_TEST(self):
        return self._config.get('Save', 'eval_test')

    @property
    def PRED_TEST(self):
        return self._config.get('Save', 'pred_test')

    @property
    def BEST_MODEL(self):
        return self._config.get('Save', 'best_model')

    @property
    def LAST_MODEL(self):
        return self._config.get('Save', 'last_model')

    @property
    def STATISTIC_FILE(self):
        return self._config.get('Save', 'statistic_file')    

    @property
    def MODEL_FILE(self):
        return self._config.get('Load', 'model_file')

    @property   
    def N_EPOCHS(self):
        return self._config.getint('Run', 'epochs')

    @property   
    def TRAIN_BATCH_SIZE(self):
        return self._config.getint('Run', 'train_batch_size')

    @property   
    def TEST_BATCH_SIZE(self):
        return self._config.getint('Run', 'test_batch_size')

    @property   
    def DYNET_MEM(self):
        return self._config.getint('Dynet', 'dynet_mem')

    @property   
    def DYNET_SEED(self):
        return self._config.getint('Dynet', 'dynet_seed')  

    @property   
    def DYNET_AUTOBATCH(self):
        return self._config.getboolean('Dynet', 'dynet_autobatch') 

    
class SeqHeadSelConfigurator(BaseConfigurator):

    def __init__(self, config_file, extra_args):
        config = SafeConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        super(SeqHeadSelConfigurator, self).__init__(config)
        
        config.write(open(config_file, 'w'))

        print 'Loaded config file sucessfully.'
        for section in config.sections():
            for k, v in config.items(section):
                print k, v
    
    @property
    def WORD_DIM(self):
        return self._config.getint('Network', 'word_dims')

    @property
    def UPOS_DIM(self):
        return self._config.getint('Network', 'upos_dims')

    @property
    def XPOS_DIM(self):
        return self._config.getint('Network', 'xpos_dims')

    @property
    def REL_DIM(self):
        return self._config.getint('Network', 'rel_dims')

    @property
    def CHAR_DIM(self):
        return self._config.getint('Network', 'char_dims')

    @property
    def MAXN_CHAR(self):
        return self._config.getint('Network', 'maxn_char')

    @property
    def N_FILTER(self):
        return self._config.getint('Network', 'n_filter')

    @property
    def WIN_SIZES(self):
        win_sizes = self._config.get('Network', 'win_sizes')
        return [int(size) for size in win_sizes.split('_')]

    @property
    def CNN_ACT(self):
        return getattr(dy, self._config.get('Network', 'cnn_act'))    

    @property
    def ENC_LAYERS(self):
        return self._config.getint('Network', 'enc_layers')

    @property
    def ENC_H_DIM(self):
        return self._config.getint('Network', 'enc_dims')

    @property
    def DEC_LAYERS(self):
        return self._config.getint('Network', 'dec_layers')

    @property
    def DEC_H_DIM(self):
        return self.ENC_H_DIM * 2

    @property
    def DEC_X_DIM(self):
        return self.DEC_H_DIM*3 + self.REL_DIM

    @property
    def LEARNING_RATE(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def ADAM_BETA1(self):
        return self._config.getfloat('Optimizer', 'adam_beta1')

    @property
    def ADAM_BETA2(self):
        return self._config.getfloat('Optimizer', 'adam_beta2')

class BiaffineConfigurator(BaseConfigurator):

    def __init__(self, config_file, extra_args):
        config = SafeConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        super(BiaffineConfigurator, self).__init__(config)
        
        config.write(open(config_file, 'w'))

        print 'Loaded config file sucessfully.'
        for section in config.sections():
            for k, v in config.items(section):
                print k, v

    @property
    def WORD_DIM(self):
        return self._config.getint('Network', 'word_dims')

    @property
    def UPOS_DIM(self):
        return self._config.getint('Network', 'upos_dims')

    @property
    def XPOS_DIM(self):
        return self._config.getint('Network', 'xpos_dims')

    @property
    def REL_DIM(self):
        return self._config.getint('Network', 'rel_dims')

    @property
    def CHAR_DIM(self):
        return self._config.getint('Network', 'char_dims')

    @property
    def EMB_DROPOUT(self):
        return self._config.getfloat('Network', 'emb_dropout')

    @property
    def MAXN_CHAR(self):
        return self._config.getint('Network', 'maxn_char')

    @property
    def N_FILTER(self):
        return self._config.getint('Network', 'n_filter')

    @property
    def WIN_SIZES(self):
        win_sizes = self._config.get('Network', 'win_sizes')
        return [int(size) for size in win_sizes.split('_')]

    @property
    def CNN_ACT(self):
        return getattr(dy, self._config.get('Network', 'cnn_act'))    

    @property
    def ENC_LAYERS(self):
        return self._config.getint('Network', 'enc_layers')

    @property
    def ENC_H_DIM(self):
        return self._config.getint('Network', 'enc_dims')

    @property
    def MLP_ARC_SIZE(self):
        return self._config.getint('Network', 'mlp_arc_size')

    @property
    def MLP_SIZES(self):
        mlp_sizes = self._config.get('Network', 'mlp_sizes')
        return [int(size) for size in mlp_sizes.split('_')]

    @property
    def MLP_LAYERS(self):
        mlp_sizes = self._config.get('Network', 'mlp_sizes')
        return len(mlp_sizes.split('_'))

    @property
    def MLP_ACT(self):
        return getattr(dy, self._config.get('Network', 'mlp_act'))

    @property
    def MLP_DROPOUT(self):
        return self._config.getfloat('Network', 'mlp_dropout')

    @property
    def RNN_H_DROPOUT(self):
        return self._config.getfloat('Network', 'rnn_h_dropout')

    @property
    def RNN_X_DROPOUT(self):
        return self._config.getfloat('Network', 'rnn_x_dropout')

    @property
    def LEARNING_RATE(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def ADAM_BETA1(self):
        return self._config.getfloat('Optimizer', 'adam_beta1')

    @property
    def ADAM_BETA2(self):
        return self._config.getfloat('Optimizer', 'adam_beta2')
        
class ChainHeadSelConfigurator(BaseConfigurator):

    def __init__(self, config_file, extra_args):
        config = SafeConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict(
                [(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        super(ChainHeadSelConfigurator, self).__init__(config)
        
        config.write(open(config_file, 'w'))

        print 'Loaded config file sucessfully.'
        for section in config.sections():
            for k, v in config.items(section):
                print k, v
    
    @property
    def WORD_DIM(self):
        return self._config.getint('Network', 'word_dims')

    @property
    def UPOS_DIM(self):
        return self._config.getint('Network', 'upos_dims')

    @property
    def XPOS_DIM(self):
        return self._config.getint('Network', 'xpos_dims')

    @property
    def REL_DIM(self):
        return self._config.getint('Network', 'rel_dims')

    @property
    def CHAR_DIM(self):
        return self._config.getint('Network', 'char_dims')

    @property
    def MAXN_CHAR(self):
        return self._config.getint('Network', 'maxn_char')

    @property
    def N_FILTER(self):
        return self._config.getint('Network', 'n_filter')

    @property
    def WIN_SIZES(self):
        win_sizes = self._config.get('Network', 'win_sizes')
        return [int(size) for size in win_sizes.split('_')]

    @property
    def RNN_H_DIM(self):
        return self._config.getint('Network', 'rnn_h_dims')

    @property
    def RNN_LAYERS(self):
        return self._config.getint('Network', 'rnn_layers')

    @property
    def MLP_SIZES(self):
        mlp_sizes = self._config.get('Network', 'mlp_sizes')
        return [int(size) for size in mlp_sizes.split('_')]

    @property
    def MLP_LAYERS(self):
        mlp_sizes = self._config.get('Network', 'mlp_sizes')
        return len(mlp_sizes.split('_'))

    @property
    def MLP_ACT(self):
        return self._config.get('Network', 'mlp_act')

    @property
    def MLP_DROPOUT(self):
        return self._config.getfloat('Network', 'mlp_dropout')

    @property
    def LEARNING_RATE(self):
        return self._config.getfloat('Optimizer', 'learning_rate')


class HeadAutomataConfigurator(BaseConfigurator):

    def __init__(self, config_file, extra_args):
        config = SafeConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict(
                [(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        super(HeadAutomataConfigurator, self).__init__(config)
        
        config.write(open(config_file, 'w'))

        print 'Loaded config file sucessfully.'
        for section in config.sections():
            for k, v in config.items(section):
                print k, v
    
    @property
    def WORD_DIM(self):
        return self._config.getint('Network', 'word_dims')

    @property
    def UPOS_DIM(self):
        return self._config.getint('Network', 'upos_dims')

    @property
    def XPOS_DIM(self):
        return self._config.getint('Network', 'xpos_dims')

    @property
    def REL_DIM(self):
        return self._config.getint('Network', 'rel_dims')

    @property
    def CHAR_DIM(self):
        return self._config.getint('Network', 'char_dims')

    @property
    def MAXN_CHAR(self):
        return self._config.getint('Network', 'maxn_char')

    @property
    def N_FILTER(self):
        return self._config.getint('Network', 'n_filter')

    @property
    def WIN_SIZES(self):
        win_sizes = self._config.get('Network', 'win_sizes')
        return [int(size) for size in win_sizes.split('_')]

    @property
    def RNN_H_DIM(self):
        return self._config.getint('Network', 'rnn_h_dims')

    @property
    def RNN_LAYERS(self):
        return self._config.getint('Network', 'rnn_layers')

    @property
    def MLP_SIZES(self):
        mlp_sizes = self._config.get('Network', 'mlp_sizes')
        return [int(size) for size in mlp_sizes.split('_')]

    @property
    def MLP_LAYERS(self):
        mlp_sizes = self._config.get('Network', 'mlp_sizes')
        return len(mlp_sizes.split('_'))

    @property
    def MLP_ACT(self):
        return self._config.get('Network', 'mlp_act')

    @property
    def MLP_DROPOUT(self):
        return self._config.getfloat('Network', 'mlp_dropout')

    @property
    def LEARNING_RATE(self):
        return self._config.getfloat('Optimizer', 'learning_rate')


class BiaffineHeadAutomataConfigurator(BaseConfigurator):

    def __init__(self, config_file, extra_args):
        config = SafeConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict(
                [(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        super(BiaffineHeadAutomataConfigurator, self).__init__(config)
        
        config.write(open(config_file, 'w'))

        print 'Loaded config file sucessfully.'
        for section in config.sections():
            for k, v in config.items(section):
                print k, v
    
    @property
    def WORD_DIM(self):
        return self._config.getint('Network', 'word_dims')

    @property
    def UPOS_DIM(self):
        return self._config.getint('Network', 'upos_dims')

    @property
    def XPOS_DIM(self):
        return self._config.getint('Network', 'xpos_dims')

    @property
    def REL_DIM(self):
        return self._config.getint('Network', 'rel_dims')

    @property
    def CHAR_DIM(self):
        return self._config.getint('Network', 'char_dims')

    @property
    def MAXN_CHAR(self):
        return self._config.getint('Network', 'maxn_char')

    @property
    def N_FILTER(self):
        return self._config.getint('Network', 'n_filter')

    @property
    def WIN_SIZES(self):
        win_sizes = self._config.get('Network', 'win_sizes')
        return [int(size) for size in win_sizes.split('_')]

    @property
    def ENC_H_DIM(self):
        return self._config.getint('Network', 'enc_h_dims')

    @property
    def ENC_LAYERS(self):
        return self._config.getint('Network', 'enc_layers')

    @property
    def DEC_H_DIM(self):
        return self._config.getint('Network', 'dec_h_dims')

    @property
    def DEC_LAYERS(self):
        return self._config.getint('Network', 'dec_layers')

    @property
    def MLP_SIZES(self):
        mlp_sizes = self._config.get('Network', 'mlp_sizes')
        return [int(size) for size in mlp_sizes.split('_')]

    @property
    def MLP_LAYERS(self):
        mlp_sizes = self._config.get('Network', 'mlp_sizes')
        return len(mlp_sizes.split('_'))

    @property
    def MLP_ACT(self):
        return self._config.get('Network', 'mlp_act')

    @property
    def MLP_DROPOUT(self):
        return self._config.getfloat('Network', 'mlp_dropout')

    @property
    def LEARNING_RATE(self):
        return self._config.getfloat('Optimizer', 'learning_rate')



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/default1.cfg')
    args, extra_args = argparser.parse_known_args()
    print args, extra_args

    config = Configurable(args.config_file, extra_args)
    print config

