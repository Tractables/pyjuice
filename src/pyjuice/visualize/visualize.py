"""
@Author: Bonan Yan

@Function: Visualization of probabilistic circuits

@Dependency: matplotlib, networkx

@Usage: 
import pyjuice.visualize as juice_vis

# default topology plot: need plt.show() or plt.savefig() following
juice_vis.plot_pc(root_node) 

# display node_num and node_id: need plt.show() or plt.savefig() following
juice_vis.plot_pc(root_node, node_id = True, node_num_label = True) 

"""
import matplotlib.pyplot as plt
import networkx as nx
from pyjuice.io.serialization import serialize_nodes
import numpy as np


def plot_pc(ns, 
           node_id : bool                 = False,
           node_num_label :bool          = False, 
           label_font_size : int         = 8,
           node_num_label_offset : int    = 1,
           node_id_offset : int          = 1) :
    """
    ns: root node
    depend: networkx, matplotlib
    """
    G = nx.DiGraph()
    node_list = serialize_nodes(ns)
    pos = {}
    nx.set_node_attributes(G, [], "node_type")
    nx.set_node_attributes(G, [], "num_nodes")
    nx.set_node_attributes(G, [], "node_id")

    for i_node in range(len(node_list)):
        G.add_node(i_node)
        
        for j in node_list[i_node]['chs']:
            G.add_edge(j,i_node)
        
        if node_list[i_node]['type']=='Input':
            G.nodes[i_node]['node_type'] = '◯'
        elif node_list[i_node]['type']=='Product':
            G.nodes[i_node]['node_type'] = '×'
        elif node_list[i_node]['type']=='Sum':
            G.nodes[i_node]['node_type'] = '+'
        else:
            G.nodes[i_node]['node_type'] = 'TypeWrong'
        
        G.nodes[i_node]['num_nodes'] = node_list[i_node]['num_nodes']
        G.nodes[i_node]['node_id'] = i_node

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    # pos = nx.nx_agraph.graphviz_layout(G, prog="fdp")
    labels = nx.get_node_attributes(G, 'node_type') 
    options = {
        "with_labels"   : True, 
        "labels"        : labels,
        "node_color"    : "#FFFFFF",
        "font_color"    : "#000000",
        "width"         : 2.0, #note: every cluster of edges is a group of edges, often sparse connection
        "edgecolors"    : "black",
    }
    pos = {node: (x,-y) for (node, (x,y)) in pos.items()}
    nx.draw(G, pos, **options)
    
    if node_num_label:
        attr_num_nodes = nx.get_node_attributes(G, 'num_nodes')
        pos_attrs = {}
        for node, coords in pos.items():
            pos_attrs[node] = (coords[0] + node_num_label_offset, coords[1] + node_num_label_offset)
        nx.draw_networkx_labels(G, 
                                pos_attrs, 
                                font_color = '#E6756F',
                                font_size = label_font_size,
                                labels = attr_num_nodes,
                                verticalalignment = 'top',
                                horizontalalignment = 'left')
    
    if node_id:
        attr_num_nodes = nx.get_node_attributes(G, 'node_id')
        pos_attrs_id = {}
        for node, coords in pos.items():
            pos_attrs_id[node] = (coords[0] + node_id_offset, coords[1] + node_id_offset)
        nx.draw_networkx_labels(G, 
                                pos_attrs_id, 
                                font_color = '#7F7DB5',
                                font_weight='bold',
                                font_size = label_font_size,
                                labels = attr_num_nodes,
                                verticalalignment = 'bottom',
                                horizontalalignment = 'left')
    return


def plot_tensor_node_connection(ns, node_id : int = 0):
    G = nx.DiGraph()
    node_list = serialize_nodes(ns)
    node_target = node_list[node_id]
    pos = {}
    
    if node_target['type']=='Input':
        # print(f'The target node {node_id} is a Input node...')
        # input_node_list = []
        # for i in range(node_target['num_nodes']):
        #     G.add_node(f'i{i}')
        #     input_node_list.append(f'i{i}')
        # pos = nx.circular_layout(G)
        # nx.draw(G, pos, node_color="#A5CE9D", with_labels=True, font_size=6) # green is sum nodes
        # return input_node_list #return input node list
        print(f"\n>>The target node {node_id} is a Input node, it has {node_target['num_nodes']} nodes, & no connection among them.<<\n")
        return
    
    elif node_target['type']=='Product':
        print(f"\n>>The target node {node_id} is a Product node")
        print(f">> Num_nodes: {node_target['num_nodes']}")
        print(f">> Child num_nodes: {node_list[node_target['chs'][0]]['num_nodes']}\n")
        return np.array(node_target['edge_ids'])
        
    elif node_target['type']=='Sum':
        print(f'The target node [{node_id}] is a Sum node...')
        # add nodes
        sum_node_list = []
        prod_node_list = []
        node_target_prd_chs = node_target['chs'][0] # integer
        adjacency_manual = np.zeros((node_target['num_nodes'], node_list[node_target_prd_chs]['num_nodes']))
        # left, sum: 
        for i in range(node_target['num_nodes']):
            G.add_node(f's{i}')
            sum_node_list.append(f's{i}')
        # right, product:
        for i in range(node_list[node_target_prd_chs]['num_nodes']):
            G.add_node(f'p{i}')
            prod_node_list.append(f'p{i}')
        
        # add edges
        _,c = node_target['edge_ids'].shape
        for i in range(c):
            G.add_edge(f"s{node_target['edge_ids'][0,i]}",f"p{node_target['edge_ids'][1,i]}") 
            adjacency_manual[node_target['edge_ids'][0,i],node_target['edge_ids'][1,i]] = 1
        
        # graph_plot
        # print('>>[Progress]: calculating graph layout...<<')
        # # pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        # pos = nx.circular_layout(G)
        # print('>>[Progress]: calculated done!<<')
        # options = {
        #     'node_size': 5,
        # }
        # print('>>[Progress]: drawing...<<')
        # nx.draw_networkx_nodes(G, pos, nodelist=sum_node_list, node_color="#A5CE9D", **options) # green is sum nodes
        # nx.draw_networkx_nodes(G, pos, nodelist=prod_node_list, node_color="#A2C8DD", **options) # blue is product nodes
        # nx.draw_networkx_edges(G, pos, width=0.1)
        
        #adjacency matrix heatmap plot
        fig, ax = plt.subplots()
        ax.spy(adjacency_manual)
        ax.set_aspect('auto')
        ax.set_xlabel('product nodes id')
        ax.set_ylabel('sum nodes id')
        ax.set_title(f"connection adjacency matrix of sum node {node_id} to its child")
        return adjacency_manual # return adjacency matrix
    else:
        print('--Plot Node Type Wrong!--')
        return False
            