
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(30)
np.random.seed(30)

def Bipartite_Maximal(N,M,P):

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

			n_nodos = N + M

			# Lista de todas las aristas del grafo
			Aristas = list(G.edges())

			# Set de nodos vacío
			Set_nodos = set()

			# Set de aristas vacío
			Set_aristas = set()

			# Realiza hasta que la lista de aristas se quede vacia
			while len(Aristas) != 0:

				arista = Aristas.pop()

				# si ninguno de los nodos está en el conjunto: agregalo
				if arista[0] not in Set_nodos and arista[1] not in Set_nodos:

					# Añadiendo al conjunto de vértices
					Set_nodos.add(arista[0])

					# Añadiendo al conjunto de vértices
					Set_nodos.add(arista[1])

					# Añadiendo la nueva arista
					Set_aristas.add(arista)

					# Print de la solución
			print(Set_aristas)
			return G, Set_aristas


N , M = 4, 4

prob = 0.3

Grafico,Set = Bipartite_Maximal(N, M, prob)

pos = nx.bipartite_layout(Grafico, list(range(0, N)))

edge_colors = ['red' if edge in Set else 'black' for edge in Grafico.edges]

nx.draw(Grafico, pos, nodelist = list(range(0, N)), node_color="red", with_labels = True )

nx.draw(Grafico, pos, nodelist = list(range(N, N + M)), node_color="blue", with_labels = True)

nx.draw_networkx_edges(Grafico, pos, edgelist=Grafico.edges(), edge_color=edge_colors, width=2)

plt.show()