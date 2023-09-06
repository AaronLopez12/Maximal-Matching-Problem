import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(12)
random.seed(12)

def Bipartite_graph(N,M,P):
		
	while True:
		plt.clf()
		G = nx.Graph()
		G.add_nodes_from(range(0, N), bipartite = 0)
		G.add_nodes_from(range(N, N + M ), bipartite = 1)

		edges = []

		for arista_A in range(0, N):
			for arista_B in range(N, N + M):

				if np.random.uniform() >  1 - P:
					edges.append((arista_A, arista_B)) 
		
		G.add_edges_from(edges)


		
		if nx.is_connected(G) == True:

			best = 0
			total_aristas = G.number_of_edges()
			n_nodos = N + M

			for option in range(1, (2**n_nodos) + 1):

				vector = np.array([int(bit) for bit in str(bin(option)[2:]).zfill(N + M)])

				tmp = set()

				tmp_2 = set()

				Vertices_incluidos = np.where(vector == 1)[0]

				for vertice in Vertices_incluidos:

					vertice = N + M - 1 - vertice

					for aristas in G.edges():

						if vertice in aristas:

							tmp.add(aristas)

							tmp_2.add(vertice)

				if len(tmp) == total_aristas and len(tmp_2) < n_nodos:
					best = np.copy(vector)
					n_nodos = len(tmp_2)

			print(best)
			return G
		
			
N , M = 4, 4
prob = 0.3
Grafico = Bipartite_graph(N, M, prob)

pos = nx.bipartite_layout(Grafico, list(range(0, N)))
nx.draw(Grafico, pos, nodelist = list(range(0, N)), node_color="red", with_labels = True )
nx.draw(Grafico, pos, nodelist = list(range(N, N + M)), node_color="blue", with_labels = True)
plt.show()