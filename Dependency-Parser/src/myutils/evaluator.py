import os

def evaluator(gold_file, pred_file, eval_file, criteria='PTB_conllu'):
    if criteria == 'PTB_conllu':
        os.system('perl eval/eval.pl -g %s -s %s > %s' % (
            gold_file, pred_file, eval_file))
        with open(eval_file, 'r') as fp:
            LAS = float(fp.readline().split()[-2])
            UAS = float(fp.readline().split()[-2])
        return LAS, UAS


if __name__ == '__main__':
    pass