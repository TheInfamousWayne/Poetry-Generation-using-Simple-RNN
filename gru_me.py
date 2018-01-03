import numpy as np
import theano
import theano.tensor as T

from utils import init_weight


class GRU:
    def __init__(self, Mi, Mo, activation):
        self.Mi = Mi
        self.Mo = Mo
        self.f = activation

        Wxh = init_weight(Mi,Mo)
        Whh = init_weight(Mo,Mo)
        bh = np.zeros(Mo)
        h0 = np.zeros(Mo)
        Wxr = init_weight(Mi,Mo)
        Whr = init_weight(Mo,Mo)
        br = np.zeros(Mo)
        Wxz = init_weight(Mi,Mo)
        Whz = init_weight(Mi,Mo)
        bz = np.zeros(Mo)

        self.Wxh = theano.shared(Wxh)
        self.Whh = theano.shared(Whh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wxr = theano.shared(Wxr)
        self.Whr = theano.shared(Whr)
        self.br = theano.shared(bh)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)
        self.bz = theano.shared(bh)
        self.params = [self.Wxh, self.Whh, self.bh, self.h0, self.Wxr, self.Whr, self.br, self.Wxz, self.Whz, self.bz]

    def recurrence(self, x_t, h_t1):
        r = T.nnet.sigmoid(x_t.dot(self.Wxr) + h_t1.dot(self.Whr) + self.br)
        z = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
        hhat = self.f(x_t.dot(self.Wxh) + (r * h_t1).dot(self.Whh) + self.bh)
        h = (1 - z) * h_t1 + z * hhat
        return h

    def output(self, x):
        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            n_steps=x.shape[0],
            outputs_info=[self.h0],
        )
        return h
