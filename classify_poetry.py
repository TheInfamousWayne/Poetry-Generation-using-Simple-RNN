# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 23:24:52 2017

@author: Vaibhav
"""

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from utils import init_weight, get_poetry_classifier_data

class SimpleRNN:
    def __init__ (self, M, V):
        self.M = M
        self.V = V

    def fit(self, X, Y, learning_rate=10e-1, mu=0.99, reg=1.0, epochs=500, show_fig=False, activation=T.tanh):
        M = self.M
        V = self.V
        K = len(set(Y))

        X, Y = shuffle(X, Y)
        Nvalid = 10
        Xvalid, Yvalid = X[-Nvalid:], Y[-Nvalid:]
        X, Y = X[:-Nvalid], Y[:-Nvalid]
        N = len(X)

        Wx = init_weight(V, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, K)
        bo = np.zeros(K)
        thX, thY, py_x, prediction = self.set(Wx, Wh, bh, h0, Wo, bo, activation)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]),thY]))
        grad = T.grad(cost, self.param)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        lr = T.scalar('learning_rate')

        updates = [
                (p,p + mu*dp - lr*g) for p,dp,g in zip(self.params, grad, dparams)
        ] + [
                (dp,mu*dp - lr*g) for dp,g in zip(dparams, grad)
        ]

        self.train_op = theano.function(
                inputs=[thX, thY, lr],
                outputs=[cost,prediction],
                updates=updates,
                allow_input_downcast=True,
        )



        costs = []
        for i in range(epochs):
            cost = 0
            Ncorrect = 0
            X, Y = shuffle(X,Y)
            for j in range(N):
                c, p = self.train_op(X[j], Y[j], learning_rate)
                cost += c
                if p==Y[j] :
                    Ncorrect += 1
            costs.append(cost)
            learning_rate *= 0.9999

            NVcorrect = 0
            for j in range(Nvalid):
                input = Xvalid[j]
                predict = self.predict_op(input)
                if predict==Yvalid[j] :
                    NVcorrect += 1

            print ('epoch: %d, cost: %f ,accuracy: %f' % (i,cost,Ncorrect/N))
            print ('Validation accuracy: ', NVcorrect/Nvalid)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def set(self, Wx, Wh, bh, h0, Wo, bo, activation):
        self.f = activation

        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.ivector('thX')
        thY = T.ivector('thY')

        def recurrence(x_t,h_t1):
            h_t = self.f(self.Wx[x_t] + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h,y],_ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0,None],
            sequences=thX,
            n_steps=thX.shape[0],
        )

        py_x = y[-1,0,:]
        prediction = T.argmax(py_x)

        self.predict_op = theano.function(
                inputs=[thX],
                outputs=[prediction],
                allow_input_downcast=True,
        )

        return thX, thY, py_x, prediction

    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])

    @staticmethod
    def load(self, filename, activation):
        npz = np.load(filename)
        Wx = npz['arr_0']
        Wh = npz['arr_1']
        bh = npz['arr_2']
        h0 = npz['arr_3']
        Wo = npz['arr_4']
        bo = npz['arr_5']
        V,M = Wx.shape
        rnn = SimpleRNN(M,V)
        rnn.set(Wx, Wh, bh, h0, Wo, bo, activation)
        return rnn

def train_poetry():
    X, Y, V = get_poetry_classifier_data(samples_per_class=500)
    rnn = SimpleRNN(30, V)
    rnn.fit(X, Y, learning_rate=10e-7, activation=T.nnet.relu, epochs=1000, show_fig=True)
    rnn.save('RNN_classify.npz')


if __name__ == "__main__":
    train_poetry()
