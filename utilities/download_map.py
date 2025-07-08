import osmnx as ox
import networkx as nx

def sample_largest_component(graph):
    largest_cc = max(nx.connected_components(graph), key=len)
    return graph_undirected.subgraph(largest_cc).copy()


if __name__ == '__main__':
    ox.settings.bidirectional_network_types = ['drive']

    place = "Quezon, Philippines"
    graph = ox.graph_from_place(place, network_type="drive")

    graph_undirected = ox.convert.to_undirected(graph)

    largest_connected_nodes_graph = sample_largest_component(graph_undirected)

    print(f"\nThe type of the graph object is: {type(largest_connected_nodes_graph)}")

    ox.io.save_graphml(largest_connected_nodes_graph,"./data/Quezon.graphml")