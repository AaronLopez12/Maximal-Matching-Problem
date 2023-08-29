#!/usr/bin/python

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

#random.seed(30)


#Random graph with N vertices, with probability (0,1) to connect two given vertices
def generateRandomGraph(N,P):
    G=nx.gnp_random_graph(N,P)
    while (nx.is_connected(G)==False):
        G=nx.gnp_random_graph(N,P)

    total_aristas = G.number_of_edges()
    
    # Total de nodos en el grafico
    n_nodos = len(G.nodes())

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

    return G
    

if __name__ == '__main__':

    # Generando una gráfica de 10 nodos con una probabilidad de 0.1 
    # entre las aristas
    G=generateRandomGraph(10,0.1)

    # Labels del grafico
    nx.draw(G,with_labels="true")

    plt.show()

        