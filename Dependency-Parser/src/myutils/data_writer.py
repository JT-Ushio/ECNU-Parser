
def data_writer(fn, conll_gen):
    with open(fn, 'w') as fp:
        for i, sentence in enumerate(conll_gen):
            if i: fp.write('\n')
            for tokens in sentence[1:]:
                fp.write(str(tokens) + '\n')


if __name__ == '__main__':
    pass