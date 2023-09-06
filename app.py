import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random 

from shiny import App, Inputs, Outputs, Session, render, ui




# Genera un grafo aleatorio usando NetworkX
def CV_Problem(N,P):
    random.seed(30)
    G=nx.gnp_random_graph(N,P)

    while (nx.is_connected(G)==False):
        
        G=nx.gnp_random_graph(N,P)

    best = 0
    total_aristas = G.number_of_edges()
    n_nodos = len(G.nodes())

    for option in range(1, 2**N):
        
        # Convertir el numero entero a binario en una lista de numpy
        vector = np.array([int(bit) for bit in str(bin(option)[2:]).zfill(N)])
        
        # Conjunto auxiliar: Aristas
        tmp = set() 

        # Conjunto auxiliar: Nodos
        tmp_2 = set() 

        # Vertices agregados
        Vertices_incluidos = np.where(vector == 1)[0]

        for vertice in Vertices_incluidos:

            # Consideracion para el vector inverso
            # En esta implementacion se tiene [...,3,2,1,0]
            # en lugar de [0,1,2,3,...] debido a los numeros binarios
            vertice = N - 1 - vertice  

            # Iteracion sobre las aristas del grafo
            for aristas in G.edges():

                if vertice in aristas:

                    # Agregar la arista 
                    tmp.add(aristas)

                    # Agregar el vertice
                    tmp_2.add(vertice)


        if len(tmp) == total_aristas and len(tmp_2) < n_nodos:
            best = np.copy(vector)
            n_nodos = len(tmp_2)
            
    return G, best

def Max_match(N,P):
    random.seed(30)
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

    return G, Set_aristas


def Bipartite_graph(N,M,P):
        
    while True:
        random.seed(30)
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

            return G, best


def Bipartite_Maximal(N,M,P):

    while True:
        
        random.seed(30)
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
            # print(Set_aristas)
            return G, Set_aristas

app_ui = ui.page_fluid(
    ui.column(
        10,
        {"class": "col-md-10 col-lg-8 py-5 mx-auto text-lg-center text-left"},
        ui.h1(" Cubrimiento de Vértices y Maximal Matching "),
        ui.h2(" Material de José A. López"),
        ui.h3(" CIMAT Unidad Aguascalientes")
        
    ),
    ui.column(
        10,
        {"class": "col-md-78 col-lg-5 py-4 mx-auto"},
        # Title
        ui.input_slider(
            "n",
            "Numero de nodos",
            min = 6,
            max = 15,
            value = 6,
            step = 1,
            width="100%",
        ),
        ui.p(
            {"class": "pt-4 small"},
            "(Pocos nodos mejoran la visualización de los grafos.)",
        ),
    ),
    ui.column(
        10,
        {"class": "col-md-78 col-lg-5 py-4 mx-auto"},
        # Title
        ui.input_slider(
            "p",
            "Probabilidad de arista entre nodos",
            min = 0.1,
            max = 0.9,
            value = 0.1,
            step = 0.01,
            width="100%",
        ),
        ui.p(
            {"class": "pt-4 small"},
            "(Valores bajos para la probabilidad mejoran la visualización de los grafos.)",
        ),
    ),
    ui.panel_main(
        ui.output_plot("p1"),
    ),
    ui.column(
        10,
        {"class": "col-md-10 col-lg-8 py-10 mx-auto text-lg-center text-left"},
        ui.h1(" Cubrimiento de Vértices y Maximal Matching: "),
        ui.h2(" graficas bipartitas"),
    ),
    ui.column(
        10,
        {"class": "col-md-78 col-lg-5 py-4 mx-auto"},
        # Title
        ui.input_slider(
            "n_2",
            "Numero N de nodos del primer conjunto",
            min = 3,
            max = 7,
            value = 3,
            step = 1,
            width="100%",
        ),
    ),
        ui.column(
        10,
        {"class": "col-md-78 col-lg-5 py-4 mx-auto"},
        # Title
        ui.input_slider(
            "m_2",
            "Numero M de nodos del segundo conjunto",
            min = 3,
            max = 7,
            value = 3,
            step = 1,
            width="100%",
        ),
        ui.p(
            {"class": "pt-4 small"},
            "(Pocos nodos mejoran la visualización de los grafos.)",
        ),
    ),
    ui.column(
        10,
        {"class": "col-md-78 col-lg-5 py-4 mx-auto"},
        # Title
        ui.input_slider(
            "p_2",
            "Probabilidad de arista entre nodos",
            min = 0.1,
            max = 0.9,
            value = 0.1,
            step = 0.01,
            width="100%",
        ),
        ui.p(
            {"class": "pt-4 small"},
            "(Valores bajos para la probabilidad mejoran la visualización de los grafos.)",
        ),
    ),
    ui.panel_main(
        ui.output_plot("p2"),
    )
)



def server(input: Inputs, output: Outputs, session: Session):
    @output
    @render.plot
    def p1():

        N = int(input.n())
        P = float(input.p())  # Probabilidad de conexión entre nodos
        G_1, best = CV_Problem(N, P)
        pos_1 = nx.spring_layout(G_1)  # Posiciones de los nodos en el gráfico
        node_colors = ['red' if best[i] == 1 else '#1f79b5' for i in range(-1, -len(best) - 1, -1)]

        G_2, Set_aristas = Max_match(N, P)
        pos_2 = nx.spring_layout(G_2)  # Posiciones de los nodos en el gráfico
        edge_colors = ['red' if edge in Set_aristas else 'black' for edge in G_2.edges]
    
        fig, axs = plt.subplots(1, 2, figsize = (10,5))
        nx.draw(
            G_1,
            pos_1,
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            font_size=10,
            ax=axs[0],
        )

        nx.draw(
            G_2,
            pos_2,
            with_labels=True,
            node_size=500,
            font_size=10,
            font_color='black',
            edge_color=edge_colors,
            ax = axs[1]
        )
        return fig
    @output
    @render.plot
    def p2():

        N = int(input.n_2())
        M = int(input.m_2())
        P = float(input.p_2())  # Probabilidad de conexión entre nodos
        G_12, best = Bipartite_graph(N,M,P)
        node_colors = ['red' if best[i] == 1 else '#1f79b5' for i in range(-1, -len(best) - 1, -1)]
        pos_12 = nx.bipartite_layout(G_12, list(range(0, N)))
        nx.draw(G_12, pos_12, nodelist = list(range(0, N)), node_color=node_colors[0:N], with_labels = True )
        nx.draw(G_12, pos_12, nodelist = list(range(N, N + M)), node_color=node_colors[N:], with_labels = True)
        

        
        G_22, Set = Bipartite_Maximal(N,M,P)
        pos_22 = nx.bipartite_layout(G_22, list(range(0, N)))
        edge_colors = ['red' if edge in Set else 'black' for edge in G_22.edges]
        #nx.draw(G_22, pos_22, nodelist = list(range(0, N)), node_color="red", with_labels = True )
        #nx.draw(G_22, pos_22, nodelist = list(range(N, N + M)), node_color='#1f79b5', with_labels = True)
        nx.draw_networkx_edges(G_22, pos_22, edgelist=G_22.edges(), edge_color=edge_colors, width=2)

    
        fig, axs = plt.subplots(1, 2, figsize = (10,5))
        nx.draw(
            G_12,
            pos_12,
            nodelist = list(range(0, N)), 
            node_color=node_colors[0:N], 
            with_labels = True,
            node_size=500,
            font_size=10,
            ax = axs[0]
        )

        nx.draw(
            G_12,
            pos_12,
            nodelist = list(range(N, N + M)), 
            node_color=node_colors[N:], 
            with_labels = True,
            node_size=500,
            font_size=10,
            ax = axs[0]
        )


        nx.draw(
            G_22,
            pos_22,
            with_labels=True,
            node_size=500,
            font_size=10,
            font_color='black',
            edge_color=edge_colors,
            ax = axs[1]
        )
        return fig




app = App(app_ui, server)
