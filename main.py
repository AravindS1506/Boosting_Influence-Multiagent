import networkx as nx
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as patches

def dfs_find_back_edges(G, node, visited, rec_stack, back_edges,target):
    """
    Identify the back edges in a DFS tree with the root node as target. 
    Each back edge that does not point to the target is a part of a cycle. Thus this edge can be removed.
    Randomization is performed on each iteration of dfs for the neighbours so that a different DFS tree is obtained, 
    one that might give lower number of edges to be removed. 
    """
    visited.add(node)
    rec_stack.add(node)
    neighbors = list(G.neighbors(node))
    random.shuffle(neighbors)
    for neighbor in neighbors:
        if neighbor not in visited:
            dfs_find_back_edges(G, neighbor, visited, rec_stack, back_edges,target)
        elif neighbor in rec_stack:
            # This edge (node -> neighbor) is a back edge
            if(neighbor!=target):
                back_edges.append((node, neighbor))
    rec_stack.remove(node)

def get_gs1_graph(G,target):
    """
    Computes a type Gs1 graph using randomization and DFS
    Input:
    The graph to be modified and the node which is required to be as the global communicator or the one present in all cycles
    Output:
    A graph with node target in all cycles in the graph.
    """
    mini=25 #Specify the maximum number of edges to be removed, if lower is possible the graph will return that assuming the strong connectivity of the graph is not lost
    mini_back=[] 
    G1=G.copy()  
    for i in range(0,20000): #Repeat it to perform randomization since dfs tree depends on the ordering of processing of nodes
        visited=set()
        rec_stack=set()
        back_edges=[]
        dfs_find_back_edges(G,target,visited,rec_stack,back_edges,target)
        if(mini>len(back_edges)):   
            G1=G.copy()
            count=0
            for (u,v) in back_edges:
                G1.remove_edge(u,v)
                if nx.is_strongly_connected(G1):
                    count+=1
                else:
                    G1.add_edge(u, v)
            if(count==len(back_edges)):
                mini=count
                mini_back=back_edges
    back_edges=mini_back

    G1=G.copy()
    count=0
    for (u,v) in back_edges:
        G1.remove_edge(u,v)
        if nx.is_strongly_connected(G1):
            count+=1
        else:
            G1.add_edge(u, v)
    return G1


def get_influence_centrality(G,s1,s2,s3,s4):
    """
    Obtain the influence centrality as demonstrated in the paper.
    Input: The graph G and the stubborn agents, can be modified to accomodate more agents.
    Output: The influence of each node in the graph
    """
    adj_matrix=nx.adjacency_matrix(G)
    adj_matrix_dense = adj_matrix.todense()

    #Normalize the weights to sum to 1
    adj=np.array(adj_matrix_dense)
    in_degree=adj.sum(axis=0)
    adj=adj/in_degree

    adj=adj.T
    length=adj.shape[0]
    stub=np.zeros((length,1))
    #Specify the stubborness of each agent below
    stub[s1-1,0]=0.5
    stub[s2-1,0]=0.1
    stub[s3-1,0]=0.5
    stub[s4-1,0]=0.38
    beta=np.diag(stub.flatten())
    P=np.dot(np.linalg.inv(np.eye(length)-np.dot((np.eye(length)-beta),adj)),beta)
    return np.dot(np.ones(length).T,P)/length

def get_regions(G,max_node):
    """
    Obtain the regions with respect to the global communicator defined as max_node using topological sorting.
    Input: Graph and the global communicator
    Output: A dictionary with each node and its respective region number 
    """
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

def dfs(G, node, visited):
    """
    A DFS code to identify the reachable nodes from a given node
    """
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

def get_neutral(G,m,s1,s2,s3,s4):
    """
    Obtain neutral nodes.Peform DFS on modified graph to check reachability of nodes from stubborn agents and classify accordingly
    Input: Global communicator and stubborn agents
    Output: Classification of nodes into neutral and non-neutral agents
    """

    G1=G.copy()
    n=G1.number_of_nodes()
    
    nodes_in_s1=np.zeros((n,1))
    nodes_in_s2=np.zeros((n,1))
    nodes_in_s3=np.zeros((n,1))
    nodes_in_s4=np.zeros((n,1))

    for u in list(G.predecessors(m)):
        G1.remove_edge(u, m)

    dfs(G1,s1,nodes_in_s1)
    dfs(G1,s2,nodes_in_s2)
    dfs(G1,s3,nodes_in_s3)
    dfs(G1,s4,nodes_in_s4)

    classification={}
    for i in range(0,n):
        if(nodes_in_s1[i]==1 or nodes_in_s2[i]==1 or nodes_in_s3[i]==1 or nodes_in_s4[i]==1):
            classification[i+1]=0 
        else:
            classification[i+1]=1
    classification[m]=1
    classification[s1]=0
    classification[s2]=0
    classification[s3]=0
    classification[s4]=0
    return classification

def verify_gs1(G):
    """
    Verify if the given graph is of type GS1, by identifying all the cycles and calculating the common node.
    The global communicator must be present in all if type Gs1 else it is a general graph.
    """
    cycles = list(nx.simple_cycles(G))
    print(f"Total number of cycles: {len(cycles)}")
    if cycles:
        # Convert each cycle to a set of nodes
        cycle_node_sets = [set(cycle) for cycle in cycles]
        
        # Find the intersection of all cycle node sets
        common_nodes = set.intersection(*cycle_node_sets)
        
        if common_nodes:
            print(f"Nodes present in all cycles: {common_nodes}")
        else:
            print("There is no single node present in all cycles.")
    else:
        print("No cycles found in the graph.")

def get_edge_weight(G,a,b):
    """
    Function to return normalized edge weight (a,b) in a graph G
    """
    adj_matrix=nx.adjacency_matrix(G)
    adj_matrix_dense = adj_matrix.todense()

    adj=np.array(adj_matrix_dense)
    in_degree=adj.sum(axis=0)
    adj=adj/in_degree
    return adj[a-1][b-1]

#Load the modified graph
with open("sampson_gs1.pkl", "rb") as f:
    G = pickle.load(f)

#Perform and store mapping of node names to numbers for ease of operations
node_list = list(G.nodes())  # Get existing node names
node_mapping = {node: i+1 for i, node in enumerate(node_list)}
G_num = nx.relabel_nodes(G, node_mapping)
reverse_mapping = {v: k for k, v in node_mapping.items()}
verify_gs1(G)

#Caculate the betweeness centrality to identify the global communicator
central=nx.betweenness_centrality(G_num)
m=max(central, key=central.get)
regions=get_regions(G_num,m)

#Store the stubborn agents
s1=node_mapping["PETER"]
s2=node_mapping["HUGH"]
s3=node_mapping['SIMPLICIUS']
s4=node_mapping['AMAND']

#Classify nodes into endorsers and neutral agents.
class_endor=get_endorser(G_num,m,s1,s2,s3,s4)
z2_endorse={node for node in class_endor if class_endor[node]==2}
z2_not_endorse={node for node in class_endor if class_endor[node]==0}
class_neutral=get_neutral(G_num,m,s1,s2,s3,s4)
neutral={node for node in class_neutral if class_neutral[node]==1}
not_neutral={node for node in class_neutral if class_neutral[node]==0}


""" 
Using the endorser nodes and the non endorser nodes we modify the edge weights of the graphs and observe its effect on 
the network. As proposed in theory, the influence increases with each such modification.
"""

print(get_influence_centrality(G_num,s1,s2,s3,s4))  # Returns the current influence of all nodes
# print(reverse_mapping[8],reverse_mapping[4],reverse_mapping[5],get_edge_weight(G_num,8,4))  #Edge weight and names of nodes in set (a,b,d) are printed here
w=G_num[5][4]['weight'] #The weight of edge (d,b) is noted
if G_num.has_edge(8, 4):    #Check if edge (a,b) exists, if yes, then increase by appropriate amount, else add an edge with that weight
    G_num[8][4]['weight']+=0.96*w
else:
    G_num.add_edge(8,4,weight=0.96*w)
G_num[5][4]['weight']-=0.96*w #Decrease weight of edge (b,d) and observe the effect of edge modification (a,b,d) on influence.


#Repeat the above procedure for any further modifications


# print(reverse_mapping[8],reverse_mapping[4],reverse_mapping[5],get_edge_weight(G_num,8,4))
print(get_influence_centrality(G_num,s1,s2,s3,s4))
# print(reverse_mapping[6],reverse_mapping[9],reverse_mapping[7],get_edge_weight(G_num,6,9))
w=G_num[7][9]['weight']
if G_num.has_edge(6, 9):
    G_num[6][9]['weight']+=0.96*w
else:
    G_num.add_edge(6,9,weight=0.96*w)
G_num[7][9]['weight']-=0.96*w
# print(reverse_mapping[6],reverse_mapping[9],reverse_mapping[7],get_edge_weight(G_num,6,9))
print(get_influence_centrality(G_num,s1,s2,s3,s4))
# print(reverse_mapping[12],reverse_mapping[9],reverse_mapping[18],get_edge_weight(G_num,12,9))
w=G_num[18][9]['weight']
if G_num.has_edge(12, 9):
    G_num[12][9]['weight']+=0.96*w
else:
    G_num.add_edge(12,9,weight=0.96*w)
G_num[18][9]['weight']-=0.96*w
# print(reverse_mapping[12],reverse_mapping[9],reverse_mapping[18],get_edge_weight(G_num,12,9))
print(get_influence_centrality(G_num,s1,s2,s3,s4))
# print(reverse_mapping[8],reverse_mapping[14],reverse_mapping[13],get_edge_weight(G_num,8,14))
w=G_num[13][14]['weight']
if G_num.has_edge(8, 14):
    G_num[8][14]['weight']+=0.96*w
else:
    G_num.add_edge(8,14,weight=0.96*w)
G_num[13][14]['weight']-=0.96*w
# print(reverse_mapping[8],reverse_mapping[14],reverse_mapping[13],get_edge_weight(G_num,8,14))
print(get_influence_centrality(G_num,s1,s2,s3,s4))
# print(reverse_mapping[8],reverse_mapping[1],reverse_mapping[5],get_edge_weight(G_num,8,1))
w=G_num[5][1]['weight']
if G_num.has_edge(8, 1):
    G_num[8][1]['weight']+=0.96*w
else:
    G_num.add_edge(8,1,weight=0.96*w)
G_num[5][1]['weight']-=0.96*w
# print(reverse_mapping[8],reverse_mapping[1],reverse_mapping[5],get_edge_weight(G_num,8,1))
print(get_influence_centrality(G_num,s1,s2,s3,s4))
# print(reverse_mapping[12],reverse_mapping[10],reverse_mapping[18],get_edge_weight(G_num,12,10))
w=G_num[18][10]['weight']
if G_num.has_edge(12, 10):
    G_num[12][10]['weight']+=0.96*w
else:
    G_num.add_edge(12,10,weight=0.96*w)
G_num[18][10]['weight']-=0.96*w
# print(reverse_mapping[12],reverse_mapping[10],reverse_mapping[18],get_edge_weight(G_num,12,10))
print(get_influence_centrality(G_num,s1,s2,s3,s4))
# print(reverse_mapping[11],reverse_mapping[13],reverse_mapping[18],get_edge_weight(G_num,11,13))


w=G_num[18][13]['weight']
if G_num.has_edge(11, 13):
    G_num[11][13]['weight']+=0.96*w
else:
    G_num.add_edge(11,13,weight=0.96*w)
G_num[18][13]['weight']-=0.96*w
# print(reverse_mapping[11],reverse_mapping[13],reverse_mapping[18],get_edge_weight(G_num,11,13))
print(get_influence_centrality(G_num,s1,s2,s3,s4))


verify_gs1(G_num)