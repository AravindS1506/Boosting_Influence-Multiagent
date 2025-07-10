import networkx as nx
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as patches

def dfs(G, node, visited):
    neighbors = list(G.neighbors(node))
    for neighbor in neighbors:
        if visited[neighbor-1]==0:
            visited[neighbor-1]=1
            dfs(G, neighbor, visited)

def get_endorser(G,m,s1,s2,s3,s4):
    """
    Obtain endorsers of node s2. Peform DFS on modified graph to check reachability of nodes from stubborn agents and classify accordingly
    Input: Global communicator and stubborn agents
    Output: Classification of nodes into endorsers of s2 and non-endorsers of s2
    """
    G1=G.copy()
    n=G1.number_of_nodes()
    
    nodes_in_s1=np.zeros((n,1))
    nodes_in_s2=np.zeros((n,1))
    nodes_in_s3=np.zeros((n,1))
    nodes_in_s4=np.zeros((n,1))
    for u in list(G.predecessors(s2)):
        G1.remove_edge(u, s2)

    dfs(G1,s1,nodes_in_s1)
    dfs(G1,s2,nodes_in_s2)
    dfs(G1,s3,nodes_in_s3)
    dfs(G1,s4,nodes_in_s4)

    classification={}
    for i in range(0,n):
        if(nodes_in_s1[i]==0 and nodes_in_s2[i]==1 and nodes_in_s3[i]==0 and nodes_in_s4[i]==0):
            classification[i+1]=2 
        else:
            classification[i+1]=0 
    classification[s2]=2
    return classification

def get_regions(G,max_node):
    G1=G.copy()
    for u in list(G.predecessors(max_node)):
        G1.remove_edge(u, max_node)
    sorted_list=list(nx.topological_sort(G1))
    distances = {node: 0 for node in sorted_list}
    #Time complexity O(m+n)
    for node in sorted_list:
        for neighbour in G1.neighbors(node):
            distances[neighbour]=max(distances[node]+1,distances[neighbour])
    return distances

# Load graph and obtain list of endorsers and regions of nodes
with open("sampson_gs1.pkl", "rb") as f:
    G = pickle.load(f)


node_list = list(G.nodes())  # Get existing node names
node_mapping = {node: i+1 for i, node in enumerate(node_list)}
G_num = nx.relabel_nodes(G, node_mapping)
reverse_mapping = {v: k for k, v in node_mapping.items()}

central=nx.betweenness_centrality(G_num)
m=max(central, key=central.get)
reg=get_regions(G_num,m)


s1=node_mapping["PETER"]
s2=node_mapping["HUGH"]
s3=node_mapping['SIMPLICIUS']
s4=node_mapping['AMAND']

class_endor=get_endorser(G_num,m,s1,s2,s3,s4)
z2_endorse={node for node in class_endor if class_endor[node]==2}

# Plot the nodes at locations according to regions from get_regions
groups = {}
for node, x in reg.items():
    groups.setdefault(x, []).append(node)

pos = {}
for x, nodes in groups.items():
    y_positions = np.linspace(-(len(nodes)-1)/2, (len(nodes)-1)/2, len(nodes))
    for node, y in zip(nodes, y_positions):
        pos[node] = (x, y)

node_stub=[s1,s2,s3,s4]
short_labels = {node: name[:3] if name[:3]!="BON" else name[:4] for node, name in reverse_mapping.items()}

node_colours = []

#Colour nodes according to endorsers,stubborn agents and normal nodes.
for node in G_num.nodes():
    if node in node_stub:
        node_colours.append("red")
    elif node in z2_endorse:
        node_colours.append("pink")
    else:
        node_colours.append("orange")




# # Draw graph without node labels
nx.draw(G_num, pos, with_labels=False, node_color=node_colours, edge_color='gray', node_size=1000, font_size=10)

# Draw edges except highlighted ones
nx.draw_networkx_edges(G_num, pos, edgelist=[e for e in G_num.edges], edge_color="gray",width=3.5,arrows=True,arrowsize=45)

for node, label in short_labels.items():
    x, y = pos[node]
    plt.text(x, y+0.15, label, fontsize=30, ha='center', color='black',weight='bold')  # Shift labels to the right

# Add additional labels below nodes
for node, name in reverse_mapping.items():
    text = None
    if name == "PETER":
        text = "s1"
    elif name == "HUGH":
        text = "s2"
    elif name == "SIMPLICIUS":
        text = "s3"
    elif name == "AMAND":
        text = "s4"
    elif name == "JOHN_BOSCO":
        text = "m"

    if text:
        x, y = pos[node]
        plt.text(x, y-0.3, text, fontsize=30, ha='center', color='black') 
plt.savefig("sampson_graph.pdf", format="pdf")
plt.show()