from __future__ import division
import numpy as np

# "gaussian" implements a normalized gaussian with
# mean vector mu and covariance matrix sigma
# in parallel on the rows of data matrix X.
# If suff_variance = None,
# then the function is implemented as usual.
# in many dimensions,
# the gaussian may be highly degenerate;
# its variances in most directions are very near 0,
# making the implementation unstable.
# in this situation,
# let 0 < suff_variance < 1;
# then only the n largest variances of the gaussian are used,
# where n is the smallest number such that
# at least suff_variance (a fraction) of variance is captured;
# the gaussian is then effectively n-dimensional.

def gaussian(X, mu, sigma, suff_variance=None):
	# usual implementation
	if not suff_variance: return np.exp(-0.5*np.sum((X-mu).dot(np.linalg.inv(sigma))*(X-mu), axis=1))/((2*np.pi)**(mu.shape[0]/2)*np.linalg.det(sigma)**0.5)
	# spectral decomposition of sigma
	Lam, P = np.linalg.eigh(sigma)
	# find rank n that captures sufficient variance
	for i in range(Lam.shape[0]):
		n = i+1
		if np.sum(Lam[-n:]) >= suff_variance*np.sum(Lam): break
	# take largest n eigenvalues
	Lam = Lam[-n:,np.newaxis]
	# take n associated eigenvectors
	P = P[:,-n:]
	# approximate rank-n determinant of sigma
	det = np.prod(Lam)
	# approximate rank-n pseudoinverse of sigma
	sigma_inv = P.dot(P.T/Lam)
	# implement effective n-dim gaussian
	return np.exp(-0.5*np.sum((X-mu).dot(sigma_inv)*(X-mu), axis=1))/((2*np.pi)**(n/2)*det**0.5)

# "off_mean" implements np.mean on N samples,
# but divides by N-1 instead of N;
# this is useful for estimating variance

def off_mean(X, axis=None):
	return np.sum(X, axis=axis)/(X.shape[axis]-1) if axis else np.sum(X)/(X.size-1)


class k_means(object):

	def __init__(self, K):
		self.K = K

	def train(self, X, max_iters=100):
		N = X.shape[0]
		self.means = X[np.random.choice(X.shape[0], self.K, replace=False),:]
		self.num_iters = 0
		clusters = self.cluster(X)
		for iter in range(max_iters):
			self.means = np.array([np.mean(X[clusters==k,:], axis=0) for k in range(self.K)])
			clusters_new = self.cluster(X)
			self.num_iters += 1
			if np.array_equal(clusters_new, clusters): break
			clusters = clusters_new
		compressed = self.means[clusters,:]
		errors = np.sum((X-compressed)**2, axis=1)
		self.reconstruction_error = np.sqrt(np.mean(errors))
		self.cluster_sizes = np.array([len(clusters[clusters==k]) for k in range(self.K)])
#		self.compressed_variance = off_mean((compressed-np.mean(compressed, axis=0, keepdims=True))**2, axis=0)
		self.compressed_variance = off_mean(np.sum((compressed-np.mean(X, axis=0, keepdims=True))**2, axis=1))
		self.max_intracluster_distance = np.sqrt(max([off_mean(errors[clusters==k]) for k in range(self.K)]))
		self.min_intercluster_distance = min([min([np.linalg.norm(self.means[k,:]-self.means[l,:], ord=2) for l in range(k+1, self.K)]) for k in range(self.K-1)])
		self.dunn_index = self.min_intercluster_distance/self.max_intracluster_distance

#		errors = np.linalg.norm(X-self.means[clusters,:], ord=2, axis=1)
#		self.reconstruction_error = np.mean(errors)
#		self.intracluster_distances = np.array([np.mean(errors[clusters==k]) for k in range(self.K)])
#		self.intracluster_variances = np.array([np.mean(errors[clusters==k]**2) for k in range(self.K)])
#		self.intercluster_distances = np.linalg.norm(self.means[:,:,np.newaxis]-self.means.T[np.newaxis,:,:], ord=2, axis=1)
#		self.dunn_index = np.min(self.intercluster_distances)/np.max(self.intracluster_distances)

	def cluster(self, X):
		return np.argmin(np.sum((X[:,:,np.newaxis]-self.means.T[np.newaxis,:,:])**2, axis=1), axis=1)

	def compress(self, X):
		return self.means[self.cluster(X),:]


# expectation-maximization implementation of
# the gaussian mixture model for clustering

class gaussian_mixture(object):

	def __init__(self, K):
		# number of clusters
		self.K = K

# "train" trains the model on data set X.
# max_iters is used to prevent infinite loops in buggy situations.
# eps is the tolerance in the increase of the log-likelihood Q;
# an increase below eps terminates training.
# if the covariance sigma is almost singular,
# then it can help to add an offset term:
# sigma -> sigma + sigma_offset*(identity matrix),
# for small sigma_offset like 0.1.
# this is not sufficient for high-dimensional problems;
# for this, set suff_cluster_variance to something like 0.95
# (see gaussian implementation at the top).

	def train(self, X, max_iters=100, eps=10**-6, sigma_offset=0, suff_cluster_variance=None):
		self.suff_cluster_variance = suff_cluster_variance
		N, D = X.shape[0], X.shape[1]
		# train k-means on X
		model = k_means(self.K)
		model.train(X)
		# initialize memberships from trained k-means
		memberships = model.cluster(X)
		# initialize priors on clusters
		self.pi = np.array([len(memberships[memberships==k]) for k in range(self.K)])/N
		# initialize cluster means
		self.mu = np.array([np.mean(X[memberships==k,:], axis=0) for k in range(self.K)])
		# initialize cluster covariances
		self.sigma = np.array([np.cov(X[memberships==k], rowvar=False) for k in range(self.K)])
		# initialize probabilities given cluster membership
		G = np.array([gaussian(X, self.mu[k,:], self.sigma[k,:,:], suff_variance=self.suff_cluster_variance) for k in range(self.K)]).T
		# initialize log-likelihood
		Q = -10**6
		self.num_iters = 0
		for iter in range(max_iters):
			# compute probabilities (soft memberships)
			Z = self.pi*G
			Z = Z/np.sum(Z, axis=1, keepdims=True)
			# compute new priors
			self.pi = np.sum(Z, axis=0)/N
			# compute soft memberships per sample
			W = Z/np.sum(Z, axis=0)
			# compute new means
			# note the use of weights
			self.mu = np.sum(W[:,:,np.newaxis]*X[:,np.newaxis,:], axis=0)
			# compute differences between data and mean
			Y = X[:,:,np.newaxis]-self.mu.T[np.newaxis,:,:]
			# compute new covariances
			self.sigma = np.array([Y[:,:,k].T.dot(W[:,k,np.newaxis]*Y[:,:,k]) for k in range(self.K)])
			# compute gaussian probabilities given cluster membership
			# note the use of sigma_offset and suff_cluster_variance
			G = np.array([gaussian(X, self.mu[k,:], self.sigma[k,:,:]+sigma_offset*np.eye(D), suff_variance=self.suff_cluster_variance) for k in range(self.K)]).T
			# update log-likelihood
			Q_new = np.sum(Z*np.log(G))
			self.num_iters += 1
			# check for tolerable increase in log-likelihood
			if Q_new-Q < eps: break
			Q = Q_new
#		self.compressed_variance = off_mean(self.cluster_soft(X).dot(np.sum((self.mu-self.pi[np.newaxis,:].dot(self.mu))**2, axis=1, keepdims=True)))
		self.compressed_variance = off_mean(self.cluster_soft(X).dot(np.sum((self.mu-np.mean(X, axis=0, keepdims=True))**2, axis=1, keepdims=True)))
		self.intracluster_variances = np.trace(self.sigma, axis1=1, axis2=2)
		self.max_intracluster_distance = np.sqrt(np.max(self.intracluster_variances))
		self.min_intercluster_distance = min([min([np.linalg.norm(self.mu[k,:]-self.mu[l,:], ord=2) for l in range(k+1, self.K)]) for k in range(self.K-1)])
		self.dunn_index = self.min_intercluster_distance/self.max_intracluster_distance

	def cluster_soft(self, X):
		# compute soft memberships
		Z = self.pi*np.array([gaussian(X, self.mu[k,:], self.sigma[k,:,:], suff_variance=self.suff_cluster_variance) for k in range(self.K)]).T
		return Z/np.sum(Z, axis=1, keepdims=True)

	def cluster_hard(self, X):
		# compute hard memberships
		# (max soft memberships)
		return np.argmax(self.cluster_soft(X), axis=1)
