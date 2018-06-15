import argparse, cPickle, math, os, random, sys, time
random.seed(666)
sys.path.append('../../antdynet/')
from tqdm import tqdm  
import _dynet as dy
import numpy as np
np.random.seed(666)
from parser_models import HeadSelectorParser, BiaffineParser
from myutils.config import SeqHeadSelConfigurator, BiaffineConfigurator
from myutils.data_loader import DataLoader, build_dataset
from myutils.data_writer import data_writer
from myutils.evaluator import evaluator

def main():
    # Configuration file processing
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/default.cfg')
    argparser.add_argument(
        '--model', default='biaffine', 
        help='s2sHS: seq2seq-head-selection-model'
             'biaffine: stanford-biaffine-model')
    argparser.add_argument('--gpu', default='-1', help='GPU ID (-1 to cpu)')
    args, extra_args = argparser.parse_known_args()
    if args.model == 's2sHS':
        cfg = SeqHeadSelConfigurator(args.config_file, extra_args)
    elif args.model == 'biaffine':
        cfg = BiaffineConfigurator(args.config_file, extra_args)

    # DyNet setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dyparams = dy.DynetParams()
    dyparams.set_autobatch(cfg.DYNET_AUTOBATCH)
    dyparams.set_random_seed(cfg.DYNET_SEED)
    dyparams.set_mem(cfg.DYNET_MEM)
    dyparams.init()

    # Build the testset
    testset = build_dataset(cfg.TEST_FILE, cfg.MAXN_CHAR, train=False)
    pc = dy.ParameterCollection()

    # Build model                         
    if args.model == 's2sHS':
        parser = HeadSelectorParser(cfg, load_model=True)
    elif args.model == 'biaffine':
        parser = BiaffineParser(cfg, load_model=True)

    for sentence in testset:
        dy.renew_cg()
        parser.run(sentence, train=False)

    data_writer(cfg.PRED_TEST, testset)
    test_LAS, test_UAS = evaluator(cfg.TEST_FILE, cfg.PRED_TEST, cfg.EVAL_TEST, criteria='PTB_conllu')

    print('TEST LAS: %f TEST UAS: %f\n' % (test_LAS, test_UAS))

if __name__ == "__main__":
    main()