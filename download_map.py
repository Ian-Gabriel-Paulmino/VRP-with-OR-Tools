import osmnx as ox

place = "Quezon, Philippines"
graph = ox.graph_from_place(place, network_type="drive")

ox.io.save_graphml(graph,"./data/Quezon.graphml")