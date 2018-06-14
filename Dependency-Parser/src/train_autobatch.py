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
    argparser.add_argument('--continue_training', action='store_true', help='Load model Continue Training')
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

    # Build the dataset of the training process
    trainset, statistic = build_dataset(
        cfg.TRAIN_FILE, cfg.MAXN_CHAR, nonproj=True, train=True)
    devset = build_dataset(cfg.DEV_FILE, cfg.MAXN_CHAR, train=False)

    # Add <PAD> Token
    statistic['w2i']['<PAD>'] = len(statistic['w2i'])
    statistic['x2i']['<PAD>'] = len(statistic['x2i'])
    statistic['u2i']['<PAD>'] = len(statistic['u2i'])

    statistic['N_WORD'] = len(statistic['w2i'])+1
    statistic['N_XPOS'] = len(statistic['x2i'])+1
    statistic['N_UPOS'] = len(statistic['u2i'])+1
    statistic['N_CHAR'] = len(statistic['c2i'])+1
    statistic['N_REL']  = len(statistic['r2i'])
    pc = dy.ParameterCollection()
    trainer = dy.AdamTrainer(
        pc,
        alpha=cfg.LEARNING_RATE, 
        beta_1=cfg.ADAM_BETA1, 
        beta_2=cfg.ADAM_BETA2)
    BEST_DEV_LAS = BEST_DEV_UAS = 0

    # Build model                         
    if args.model == 's2sHS':
        parser = HeadSelectorParser(cfg, statistic, pc, args.continue_training)
    elif args.model == 'biaffine':
        parser = BiaffineParser(cfg, statistic, pc, args.continue_training)

    # Train model
    n_batches_train = int(math.ceil(len(trainset)*1.0/cfg.TRAIN_BATCH_SIZE))
    for epoch in xrange(cfg.N_EPOCHS):
        random.shuffle(trainset)
        loss_all_train = []
        for i in tqdm(range(n_batches_train), ncols=100):
            dy.renew_cg()
            # Create a mini batch
            start = i*cfg.TRAIN_BATCH_SIZE
            end = start + cfg.TRAIN_BATCH_SIZE
            losses = 0.
            cnt = 0
            train_batch = trainset[start:end]

            # Mini-batch training
            for sentence in train_batch:
                sent_loss = parser.run(sentence, train=True)
                losses += sent_loss
                cnt += len(sentence)-1
            #mb_loss = dy.average(losses) * 0.5
            mb_loss = losses / cnt * 0.5

            # Forward
            loss_all_train.append(mb_loss.value())
            # Backward
            mb_loss.backward()
            # Update
            trainer.update()

        for sentence in devset:
            dy.renew_cg()
            parser.run(sentence, train=False)

        data_writer(cfg.PRED_DEV, devset)
        dev_LAS, dev_UAS = evaluator(cfg.DEV_FILE, cfg.PRED_DEV, cfg.EVAL_DEV, criteria='PTB_conllu')
        
        if (dev_LAS > BEST_DEV_LAS and dev_UAS > BEST_DEV_UAS or 
            dev_LAS+dev_UAS > BEST_DEV_LAS+BEST_DEV_UAS):
            parser.save(cfg.BEST_MODEL, cfg.STATISTIC_FILE)
        parser.save(cfg.LAST_MODEL, cfg.STATISTIC_FILE)

        print('EPOCH: %d, Train Loss: %.6f  Dev LAS: %f Dev UAS: %f\n' % (
            epoch+1, np.mean(loss_all_train), dev_LAS, dev_UAS))

if __name__ == "__main__":
    main()