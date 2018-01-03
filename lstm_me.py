import theano
import theano.tensor as T
import numpy as np

from utils import init_weight

class LSTM:
    def __init__(self, Mi, Mo, activation):
        self.f = activation
        self.Mi = Mi
        self.Mo = Mo

        # x, c, i/f, o, h
        Wxc = init_weight(Mi,Mo)
        # Wcc = init_weight(Mo,Mo)   ------> NOT THIS
        Whc = init_weight(Mo,Mo)
        bc = np.zeros(Mo)
        c0 = np.zeros(Mo)
        Wxi = init_weight(Mi,Mo)
        Wci = init_weight(Mo,Mo)
        Whi = init_weight(Mo,Mo)
        bi = np.zeros(Mo)
        Wxf = init_weight(Mi,Mo)
        Wcf = init_weight(Mo,Mo)
        Whf = init_weight(Mo,Mo)
        bf = np.zeros(Mo)
        Wxo = init_weight(Mi,Mo)
        Wco = init_weight(Mo,Mo)
        Who = init_weight(Mo,Mo)
        bo = np.zeros(Mo)
        h0 = np.zeros(Mo)

        self.Wxc = theano.shared(Wxc)
        # self.Wcc = theano.shared(Wcc)   ------> NOT THIS
        self.Whc = theano.shared(Whc)
        self.bc = theano.shared(bc)
        self.c0 = theano.shared(c0)
        self.Wxi = theano.shared(Wxi)
        self.Wci = theano.shared(Wci)
        self.Whi = theano.shared(Whi)
        self.bi = theano.shared(bi)
        self.Wxf = theano.shared(Wxf)
        self.Wcf = theano.shared(Wcf)
        self.Whf = theano.shared(Whf)
        self.bf = theano.shared(bf)
        self.Wxo = theano.shared(Wxo)
        self.Wco = theano.shared(Wco)
        self.Who = theano.shared(Who)
        self.bo = theano.shared(bo)
        self.h0 = theano.shared(h0)
        self.params = [
            self.Wxi,
            self.Whi,
            self.Wci,
            self.bi,
            self.Wxf,
            self.Whf,
            self.Wcf,
            self.bf,
            self.Wxc,
            self.Whc,
            self.bc,
            self.Wxo,
            self.Who,
            self.Wco,
            self.bo,
            self.c0,
            self.h0,
        ]

    def recurrence(self, x_t, h_t1, c_t1):
        i = T.nnet.sigmoid(x_t.dot(self.Wxi) + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bc)
        f = T.nnet.sigmoid(x_t.dot(self.Wxf) + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
        c_hat = T.tanh(x_t.dot(self.Wxc) + h_t1.dot(self.Whc) + self.bc)
        c_t = f * c_t1 + i * c_hat
        o = T.nnet.sigmoid(x_t.dot(self.Wxo) + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
        h_t = o * T.tanh(c_t)
        return h_t, c_t

    def output(self, x):
        [h, c], _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=[self.h0, self.c0],
            n_steps=x.shape[0],
        )
        return h
