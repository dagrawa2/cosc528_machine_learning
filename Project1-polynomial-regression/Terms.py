import networkx as nx

class PolynomialTermsTree(nx.DiGraph):

	def __init__(self):
		nx.DiGraph.__init__(self)
		self.order = 0

	def leaves(self):
		return [N for N in self.nodes() if self.out_degree(N)==0]

	def build(self, n, k):
		for i in range(1, n+1):
			self.add_node(i, value=i)
		self.order += n
		for count in range(k-1):
			for L in self.leaves():
				for i in range(self.node[L]['value'], n+1):
					self.order += 1
					self.add_node(self.order, value=i)
					self.add_edge(L, self.order)

	def terms(self):
		terms = []
		for L in self.leaves():
			path = [L]+list(nx.ancestors(self, L))
			path.sort()
			terms.append([self.node[i]['value'] for i in path])
		return terms


def GenerateTerms(n, k):
	G = PolynomialTermsTree()
	G.build(n, k)
	return G.terms()
