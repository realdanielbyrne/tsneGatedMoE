import numpy as np
import keras.backend as K
class TSNEsp:

  def Hbeta(self,D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P) + 1e-14
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

  def x2p(self,X, u=15, tol=1e-4, print_iter=500, max_tries=50, verbose=0):
    '''
      Compute pairwise distances
    '''
    n = X.shape[0]                     # number of instances
    P = np.zeros((n, n))               # empty probability matrix
    beta = np.ones(n)                  # empty precision vector
    logU = np.log(u)                   # log of perplexity (= entropy)

    if verbose > 0: print('Computing pairwise distances...')
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X + sum_X[:,None] + -2 * X.dot(X.T)

    # Run over all datapoints
    if verbose > 0: print('Computing P-values...')
    for i in range(n):

        if verbose > 1 and print_iter and i % print_iter == 0:
            print('Computed P-values {} of {} datapoints...'.format(i, n))

        # Set minimum and maximum values for precision
        betamin = float('-inf')
        betamax = float('+inf')

        # Compute the Gaussian kernel and entropy for the current precision
        indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
        Di = D[i, indices]
        H, thisP = self.Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            H, thisP = self.Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, indices] = thisP

    if verbose > 0:
        print('Mean value of sigma: {}'.format(np.mean(np.sqrt(1 / beta))))
        print('Minimum value of sigma: {}'.format(np.min(np.sqrt(1 / beta))))
        print('Maximum value of sigma: {}'.format(np.max(np.sqrt(1 / beta))))

    return P, beta

  def compute_joint_probabilities(self, samples, batch_size, d=2, perplexity=30, tol=1e-5, verbose=0):
    '''
      Precomputes joint probabilities for all batches
    '''
    v = d - 1
    n = samples.shape[0]
    self.batch_size = min(batch_size, n)

    if verbose > 0: print('Precomputing P-values...')
    batch_count = int(n / self.batch_size )
    P = np.zeros((batch_count, self.batch_size , self.batch_size ))
    for i, start in enumerate(range(0, n - self.batch_size  + 1, self.batch_size )):
        curX = samples[start:start+self.batch_size ]                   # select batch
        P[i], beta = self.x2p(curX, perplexity, tol, verbose=verbose) # compute affinities using fixed perplexity
        P[i][np.isnan(P[i])] = 0                                 # make sure we don't have NaN's
        P[i] = (P[i] + P[i].T) # / 2                             # make symmetric
        P[i] = P[i] / P[i].sum()                                 # obtain estimation of joint probabilities
        P[i] = np.maximum(P[i], np.finfo(P[i].dtype).eps)

    return P

  def KLdivergence(self,P, Y):
    alpha = K.int_shape(Y)[1] - 1.
    sum_Y = K.sum(K.square(Y), axis=1)
    eps = K.constant(10e-15)
    D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
    Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
    Q *= K.variable(1 - np.identity(self.batch_size))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log(P) -  K.log(Q)
    C = K.sum(P * C)
    return C