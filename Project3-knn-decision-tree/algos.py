from __future__ import division
import networkx as nx
import numpy as np

class knn(object):

	def __init__(self, K):
		self.K = K

	def train(self, X, y):
		self.X_train = X
		self.y_train = y

	def p(self, X):
		return np.mean(self.y_train[np.argsort(np.sum((X[:,:,np.newaxis]-self.X_train.T[np.newaxis,:,:])**2, axis=1), axis=1)[:,:self.K]], axis=1)

	def predict(self, X):
		c = self.p(X)
		c[c<0.5] = 0
		c[c>0.5] = 1
		c[c==0.5] = np.random.randint(2, size=(len(c[c==0.5])))
		return c


def entropy(x):
	f = lambda y: -y*np.log(y) if y > 0 else 0
	p = np.mean(x)
	return f(p) + f(1-p)

def gini(x):
	p = np.mean(x)
	return 2*p*(1-p)

def mis(x):
	p = np.mean(x)
	return 1-max([p,1-p])

impurity_functions = {"entropy":entropy, "gini":gini, "mis":mis}

class decision_tree(nx.DiGraph):

	def __init__(self, K=2, eps=0.05, impurity="entropy"):
		nx.DiGraph.__init__(self)
		self.K = K
		self.eps = eps
		self.impurity = impurity_functions[impurity]
		self.add_node(0, feature=None, value=None, less=None, greater=None)
		self.order = 1

	def add_child(self, parent, branch):
		self.node[parent][branch] = self.order
		self.add_node(self.order, feature=None, value=None, less=None, greater=None)
		self.add_edge(parent, self.order, branch=branch)
		self.order += 1

	def make_leaf(self, N, p):
		self.node[N].pop("feature")
		self.node[N].pop("value")
		self.node[N].pop("less")
		self.node[N].pop("greater")
		self.node[N].update({"p":p})

	def train(self, X, y, debug=False):
		node_X = {0:X}
		node_y = {0:y}
		node_headers = {0:range(X.shape[1])}
		if debug: print "--- debug mode ---\n"
		for k in range(self.K+1):
			if debug: print "k: ", k
			nodes_to_split = node_X.keys()
			nodes_to_split.sort()
			if debug: print "nodes to split: ", nodes_to_split, "\n"
			for N in nodes_to_split:
				if debug: print "N: ", N
				N_X = node_X[N]
				N_y = node_y[N]
				N_headers = node_headers[N]
				if debug: print "N_X.shape[0]: ", N_X.shape[0]
				if debug: print "N_headers: ", N_headers
				node_X.pop(N)
				node_y.pop(N)
				node_headers.pop(N)
				if self.impurity(N_y) <= self.eps or len(N_headers) == 0 or k+1 == self.K+1:
					if debug: print "Making into a leaf; impurity is ", self.impurity(N_y)
					self.make_leaf(N, np.mean(N_y))
					if debug: print "p: ", self.node[N]["p"], "\n"
					continue
				if debug: print "not making into a leaf\n"
				impurities = []
				for j in range(len(N_headers)):
					if debug: print "j: ", j
					col_j = N_X[:,j]
					values = np.unique(col_j)
					if debug: print "values: ", values, "\n"
					impurities_j = []
					for i in range(len(values)-1):
						split_point = (values[i]+values[i+1])/2
						imp = ( self.impurity(N_y[col_j<=split_point]) + self.impurity(N_y[col_j>split_point]) )/2
						impurities_j.append([split_point, imp])
					impurities_j = np.array(impurities_j)
					i_star = np.argmin(impurities_j[:,1])
					impurities.append(impurities_j[i_star,:])
				impurities = np.array(impurities)
				j_star = np.argmin(impurities[:,1])
				if debug: print "j_star: ", j_star
				self.node[N]["feature"] = N_headers[j_star]
				self.node[N]["value"] = impurities[j_star,0]
				if debug: print "self.node["+str(N)+"]: ", self.node[N]
				col_j_star = N_X[:,j_star]
				N_X = N_X[:, np.arange(N_X.shape[1])!=j_star]
				N_headers.pop(j_star)
				if debug: print "order: ", self.order
				if debug: print "adding child"
				self.add_child(N, branch="less")
				node_X.update({self.order-1: N_X[col_j_star<=self.node[N]["value"],:]})
				node_y.update({self.order-1: N_y[col_j_star<=self.node[N]["value"]]})
				node_headers.update({self.order-1: N_headers})
				if debug: print "order: ", self.order
				if debug: print "adding child"
				self.add_child(N, branch="greater")
				if debug: print "order: ", self.order
				node_X.update({self.order-1: N_X[col_j_star>self.node[N]["value"],:]})
				node_y.update({self.order-1: N_y[col_j_star>self.node[N]["value"]]})
				node_headers.update({self.order-1: N_headers})
				if debug: print "self.node["+str(N)+"]: ", self.node[N], "\n"
		if debug: print "--- end debug ---\n"

	def p_one(self, x):
		N = 0
		for k in range(self.K+1):
			if "p" in self.node[N].keys(): break
			if x[self.node[N]["feature"]] <= self.node[N]["value"]: N = self.node[N]["less"]
			elif x[self.node[N]["feature"]] > self.node[N]["value"]: N = self.node[N]["greater"]
		return self.node[N]["p"]

	def p(self, X):
		y = []
		for i in range(X.shape[0]):
			y.append(self.p_one(X[i,:]))
		return np.array(y)

	def predict(self, X):
		c = self.p(X)
		c[c<0.5] = 0
		c[c>0.5] = 1
		c[c==0.5] = np.random.randint(2, size=(len(c[c==0.5])))
		return c

	def depth(self):
		return nx.dag_longest_path_length(self)