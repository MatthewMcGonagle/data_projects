'''
Poisson Mixture Model
'''

import numpy as np
import scipy.stats as stats

class PoissonMixtureModel:
    '''
    Assumes that the logarithm of the Poisson mean count is linear in X.
    '''

    def __init__(self, n_steps = 75, learning_rate = 0.005, n_grad_per_maximization = 20, keep_history = True, alpha = 0.0):
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.n_grad_per_maximization = n_grad_per_maximization
        self.keep_history = keep_history
        self.alpha = alpha

        self.n_poisson = 2
        self.mixture_probs = np.full(self.n_poisson, 1.0 / self.n_poisson)
        self.coefs = None
        self.intercepts = np.zeros(self.n_poisson) 

        self.history = None 

    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : Numpy array of shape (n_samples, n_features)
        y : Numpy array of shape (n_samples), dtype = int
            The counts to fit the poisson distribution to.
        '''
        _, n_features = X.shape

        self._initialize_fit_parameters(X, y)
        #self.coefs = np.random.uniform(size = (self.n_poisson, n_features))

        if self.keep_history:
            self.history = {'coefs' : [self.coefs], 'intercepts' : [self.intercepts],
                            'mixture_probs': [self.mixture_probs]}

        for i_step in range(self.n_steps):
            responsibilities = self._expectation_step(X, y)
            self._maximization_step(X, y, responsibilities)
            if self.keep_history:
                self._update_history()

    def _update_history(self):
        self.history['coefs'] = np.append(self.history['coefs'], [self.coefs], axis = 0)
        self.history['intercepts'] = np.append(self.history['intercepts'], [self.intercepts], axis = 0)
        self.history['mixture_probs'] = np.append(self.history['mixture_probs'], [self.mixture_probs], axis = 0)

    def _initialize_fit_parameters(self, X, y):
        _, n_features = X.shape 
        log_y = np.log(1 + y)
        log_y_mean = log_y.mean()
        log_y_std = log_y.std()

        self.intercepts = np.random.uniform(low = log_y_mean - 2 * log_y_std,
                                            high = log_y_mean + 2 * log_y_std,
                                            size = self.n_poisson)

        slope_guess_size = log_y_std / X.std()
        self.coefs = np.random.uniform(low = -slope_guess_size,
                                       high = slope_guess_size, 
                                       size = (self.n_poisson, n_features)) 

    def find_poisson_means(self, X):
        '''
        Returns
        -------
        means_poissons : Numpy array of shape (n_samples, n_poisson_sources)
            The means of each poisson source at each sample location.
        '''
        log_means_given_poisson = np.dot(X, self.coefs.T) + self.intercepts
        return np.exp(log_means_given_poisson)

    def find_log_prob_given_poisson(self, X, y):
        means_given_poisson = self.find_poisson_means(X)
        return stats.poisson.logpmf(y[..., np.newaxis], mu = means_given_poisson)

    def find_prob_given_poisson(self, X, y):
        '''
        Returns
        -------
        prob_given_poisson : Numpy array of shape (n_samples, n_poisson_sources)
            The conditional probabilities for each sample and each poisson source.
        '''
        means_given_poisson = self.find_poisson_means(X)

        # Make sure to add axis to y for the poisson models.
        return stats.poisson.pmf(y[..., np.newaxis], mu = means_given_poisson) 
        

    def _expectation_step(self, X, y):
        '''
        Returns
        -------
        responsibilities : Numpy array of shape (n_samples, n_poisson_sources)
            The responsibities of each sample for each poisson source, terminology coming from
            that used with the EM-algorithm. See "The Elements of Statistical Learning" by Hastie,
            Tibshirani, and Friedman.
        '''
        log_prob_given_poisson = self.find_log_prob_given_poisson(X, y)
        log_prob_given_X = log_prob_given_poisson + np.log(self.mixture_probs)
        responsibilities = log_prob_given_X - np.amax(log_prob_given_X, axis = -1, keepdims = True) 
        responsibilities = np.exp(responsibilities)
        responsibilities /= responsibilities.sum(axis = -1, keepdims = True)

        self.mixture_probs = responsibilities.sum(axis = 0)
        self.mixture_probs /= self.mixture_probs.sum()

        return responsibilities 

    def _maximization_step(self, X, y, responsibilities):
        for _ in range(self.n_grad_per_maximization):
            grad_coefs, grad_intercepts = self._find_grad(X, y, responsibilities)
            self.coefs += grad_coefs * self.learning_rate
            self.intercepts += grad_intercepts * self.learning_rate

    def _find_grad(self, X, y, responsibilities):
        '''
        Returns
        -------
        grad_coefs : Shape (n_poisson, n_features)
        grad_intercepts : Shape (n_poisson)
        '''
        # Grad_{feature_j}  = Sum_{sample_i} ( weight_i * (count_i - mean_i) 
        #                                      * X_{sample_i, feature_j} ) 

        means_given_poisson = self.find_poisson_means(X).T
        x_weights = responsibilities.T * (y - means_given_poisson) 
        totals = responsibilities.sum(axis = 0)
        grad_coefs = np.dot(x_weights, X) / totals.reshape(-1, 1) 
        grad_coefs -= self.alpha * self.coefs
        grad_intercepts = x_weights.sum(axis = -1) / totals 
        return grad_coefs, grad_intercepts
