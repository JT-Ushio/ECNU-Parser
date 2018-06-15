import _dynet as dy
import numpy as np
import sys
from attention_mechanism import VanillaAttention
from nn_classifier import BiaffineLabelClassifier
from multi_layer_perception import MLP

class BiaffineDecoder(object):
    """This builds seq2seq Head Selection Decoder:

    :param model dynet.ParameterCollection: 
    :param n_layers int: Number of LSTM layers 
    :param x_dim int: Dimension of LSTM input :math:`\\boldsymbol{x}`
    :param h_dim int: Dimension of LSTM hidden state :math:`\\boldsymbol{h}`
    :param n_rel int: Number of relation 
    :param r_dim int: Dimension of relation embedding
    :param LSTMBuilder dynet._RNNBuilder: Dynet LSTM type
    :returns: sentence loss
    :rtype: dynet.Expression
    """
    def __init__(self, model, mlp_config, rnum, biaffine_config):
        pc = model.add_subcollection()
        
        self.head_mlp = MLP(
            pc, mlp_config['sizes'], mlp_config['act'],
            mlp_config['bias'], mlp_config['dropout'])

        self.son_mlp  = MLP(
            pc, mlp_config['sizes'], mlp_config['act'],
            mlp_config['bias'], mlp_config['dropout'])
        arc_size = mlp_config['arc_size']
        rel_size = mlp_config['sizes'][-1]-mlp_config['arc_size']
        self.arc_ptr = BiaffineLabelClassifier(
            pc, arc_size, arc_size, 1,
            h_bias=biaffine_config['arc']['h_bias'], 
            s_bias=biaffine_config['arc']['s_bias'])

        self.rel_ptr = BiaffineLabelClassifier(
            pc, rel_size, rel_size, rnum,
            h_bias=biaffine_config['rel']['h_bias'], 
            s_bias=biaffine_config['rel']['s_bias'])

        self.mlp_config = mlp_config
        self.pc = pc
        self.spec = (mlp_config, rnum, biaffine_config)

    def __call__(self, sentence, h_list, r2i, i2r=None, train=False):
        slen, rnum = len(sentence), len(r2i)
        X = dy.concatenate_cols(h_list)
        # print slen, len(h_list), sentence[0]
        if train: X = dy.dropout_dim(X, 1, self.mlp_config['dropout'])
        head_vec = self.head_mlp(X, train)
        son_vec  = self.son_mlp (X, train)

        son_arc  = son_vec [:self.mlp_config['arc_size']]
        son_rel  = son_vec [self.mlp_config['arc_size']:]
        head_arc = head_vec[:self.mlp_config['arc_size']]
        head_rel = head_vec[self.mlp_config['arc_size']:]

        # e_dim: (slen, slen) [head, son]
        e_arc = self.arc_ptr(head_arc, son_arc)
        
        if train:
            # e_dim: ((slen), slen) [head]*son
            e_arc = dy.reshape(e_arc, (slen,), slen)
            golds = [t.head for t in sentence]
            golds[0] = 0 # Mask *ROOT*
            masks = dy.inputTensor(
                [0. if i==0 else 1. for i in range(slen)], batched = True)
            arc_losses = dy.pickneglogsoftmax_batch(e_arc, golds)
            arc_losses = dy.sum_batches(arc_losses*masks)
        else:
            masks = [[1. for j in range(slen)] for i in range(slen)]
            for i in range(slen): masks[i][i] = 0.
            masks = dy.inputTensor(masks)
            p_arc = np.argmax(dy.cmult(dy.softmax(e_arc), masks).npvalue(), axis=0)
            for i in range(slen-1):
                sentence[i+1].pred_head = p_arc[i+1]

        rel_dim = self.mlp_config['sizes'][-1]-self.mlp_config['arc_size']
        partial_son_rel  = dy.reshape(son_rel, (rel_dim,), slen)
        if train:
            partial_head_rel = dy.select_cols(head_rel, golds)
        else:
            p_arc[0] = 0
            partial_head_rel = dy.select_cols(head_rel, p_arc)
        partial_head_rel = dy.reshape(partial_head_rel, (rel_dim,), slen)

        e_rel = self.rel_ptr(partial_head_rel, partial_son_rel)
        if train:
            golds = [0] + [r2i[t.deprel] for t in sentence[1:]]
            masks = dy.inputTensor(
                [0. if i==0 else 1. for i in range(slen)], batched = True)
            rel_losses = dy.pickneglogsoftmax_batch(dy.transpose(e_rel), golds)
            rel_losses = dy.sum_batches(rel_losses*masks)
        else:
            e_rel = dy.reshape(dy.transpose(e_rel), (rnum, slen))
            p_rel = np.argmax(dy.softmax(e_rel).npvalue(), axis=0)
            for i in range(slen-1):
                sentence[i+1].pred_deprel = i2r[int(p_rel[i+1])]
        if train:
            return arc_losses + rel_losses

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.
        
        It is one of the prerequisites for Dynet save/load method.
        """
        mlp_config, rnum, biaffine_config = spec
        return BiaffineDecoder(model, mlp_config, rnum, biaffine_config)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.
        
        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc