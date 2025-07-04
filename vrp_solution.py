import os
import osmnx as ox
import numpy as np
import networkx as nx
import traceback
from networkx import MultiDiGraph

import folium

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


from typing import List, Union, Any


class NetworkBuilder:

    def __init__(self, number_of_nodes: int, num_vehicles: int):
        self.number_of_nodes = number_of_nodes
        self.num_vehicles = num_vehicles

        self.sampled_nodes = []  # store selected node IDs
        self.node_positions = {}  # store {node_id: (lat, lon)}

    def load_map_data(self) -> MultiDiGraph:
        return ox.io.load_graphml(filepath="./data/Quezon.graphml")

    def get_sample_nodes(self, graph) -> Any:
        number_of_nodes = 20
        nodes = list(graph.nodes)

        return np.random.choice(nodes,number_of_nodes, replace=False)
    

    def get_connected_nodes(self, graph, max_attempts=100):
        nodes = list(graph.nodes)
        
        for attempt in range(max_attempts):
            # Sample nodes
            candidate_nodes = np.random.choice(nodes, self.number_of_nodes, replace=False)
            
            # Check if all nodes are connected to all other nodes
            all_connected = True
            for i, node_i in enumerate(candidate_nodes):
                for j, node_j in enumerate(candidate_nodes):
                    if i != j:
                        try:
                            nx.shortest_path_length(G=graph, source=node_i, target=node_j, weight='length')
                        except nx.NetworkXNoPath:
                            all_connected = False
                            break
                if not all_connected:
                    break
            
            if all_connected:
                print(f"Found connected nodes on attempt {attempt + 1}")

                self.sampled_nodes = candidate_nodes
                self.node_positions = {
                    node: (graph.nodes[node]['y'], graph.nodes[node]['x']) for node in candidate_nodes
                }
                return candidate_nodes
    
        # Fallback: find largest connected component and sample from it
        print("Could not find fully connected random sample, using largest connected component")
        largest_cc = max(nx.weakly_connected_components(graph), key=len)
        largest_cc_nodes = list(largest_cc)
        
        if len(largest_cc_nodes) >= self.number_of_nodes:
            return np.random.choice(largest_cc_nodes, self.number_of_nodes, replace=False)
        else:
            print(f"Warning: Using all {len(largest_cc_nodes)} nodes from largest component")
            self.number_of_nodes = len(largest_cc_nodes)
            return np.array(largest_cc_nodes)
        

    def calculate_distance_matrix(self, graph) -> List[List[Union[int,float]]]:
        
        # Get sampled nodes from graph
        # nodes = self.get_sample_nodes(graph=graph)
        # nodes = self.get_connected_nodes(graph=graph)
        nodes = self.sampled_nodes

        distance_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes))

        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                if i == j:
                    distance_matrix[i][j] = 0
                else:
                    try:
                        length = nx.shortest_path_length(
                            G=graph,
                            source=nodes[i],
                            target=nodes[j],
                            weight='length'
                        )
                        distance_matrix[i][j] = length
                    except nx.NetworkXNoPath:
                        distance_matrix[i][j] = np.inf


        return distance_matrix
    
    def create_data_model(self) -> List[List[Union[int,float]]]:

        # 1. Load graph from data file
        graph:MultiDiGraph = self.load_map_data()
        self.sampled_nodes = self.get_connected_nodes(graph=graph)

    
        self.node_positions = {
            node: (graph.nodes[node]['y'], graph.nodes[node]['x']) for node in self.sampled_nodes
        }
        

        # 2. Convert to projected graph
        # projected_graph:MultiDiGraph = ox.project_graph(graph)
        
        # 3. Init data structure
        data = {}

        # 4. Calculate distance for each sampled node
        data['distance_matrix'] = self.calculate_distance_matrix(graph)

        # 5. Init related data
        data["num_vehicles"] = self.num_vehicles
        data["depot"] = 0
        data['graph'] = graph
        data['node_ids'] = self.sampled_nodes
        data['positions'] = self.node_positions

        return data


class VRPSolver:

    def __init__(self, data):
        self.distance_matrix = data['distance_matrix']
        self.num_vehicles = data['num_vehicles']
        self.depot = data['depot']
        self.data = data


        # routing index manager
        self.manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']), data['num_vehicles'], data['depot']
        )

        # Routing Model
        self.routing = pywrapcp.RoutingModel(self.manager)

    def distance_callback(self, from_index, to_index) -> int:
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return int(self.data['distance_matrix'][from_node][to_node])
    
    def print_solution(self, solution) -> None:
        """Prints solution on console."""
        print(f"Objective: {solution.ObjectiveValue()}")
        max_route_distance = 0
        for vehicle_id in range(self.num_vehicles):
            if not self.routing.IsVehicleUsed(solution, vehicle_id):
                continue
            index = self.routing.Start(vehicle_id)
            plan_output = f"Route for vehicle {vehicle_id}:\n"
            route_distance = 0
            while not self.routing.IsEnd(index):
                plan_output += f" {self.manager.IndexToNode(index)} -> "
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            plan_output += f"{self.manager.IndexToNode(index)}\n"
            plan_output += f"Distance of the route: {route_distance}m\n"
            print(f"These are the nodes: {plan_output}")
            max_route_distance = max(route_distance, max_route_distance)
        print(f"Maximum of the route distances: {max_route_distance}m")


    def solve(self) -> None:
        transit_callback_index = self.routing.RegisterTransitCallback(self.distance_callback)

        # Define cost of each arc
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = "Distance"
        self.routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            100000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)


        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve the problem
        try:
            solution = self.routing.SolveWithParameters(search_parameters)
            if solution:
                self.print_solution(solution)
                routes = self.get_routes(solution)
                visualize_solution(self.data, routes)
            else:
                print("No solution found (but no crash).")
        except Exception:
            traceback.print_exc()

    def get_routes(self, solution):

        routes = []

        for vehicle_id in range(self.num_vehicles):
            if not self.routing.IsVehicleUsed(solution, vehicle_id):
                continue

            index = self.routing.Start(vehicle_id)
            route = []

            while not self.routing.IsEnd(index):
                route.append(self.manager.IndexToNode(index))
                index = solution.Value(self.routing.NextVar(index))

            route.append(self.manager.IndexToNode(index))
            routes.append(route)
        
        return routes
    
def visualize_solution(data, routes):
    node_ids = data['node_ids']
    positions = data['positions']
    depot = node_ids[data['depot']]

    depot_coords = positions[depot]
    m = folium.Map(location=depot_coords, zoom_start=15)

    # Color palette
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'darkred', 'cadetblue']

    # Mark depot
    folium.Marker(
        location=depot_coords,
        popup="Depot",
        icon=folium.Icon(color="black", icon="home")
    ).add_to(m)

    # Mark delivery points
    for idx, node in enumerate(node_ids):
        if node == depot:
            continue
        folium.Marker(
            location=positions[node],
            popup=f"Node {idx}",
            icon=folium.Icon(color="gray", icon="circle")
        ).add_to(m)         

    # Draw routes
    for v_idx, route in enumerate(routes):
        route_coords = [positions[node_ids[i]] for i in route]
        color = colors[v_idx % len(colors)]
        folium.PolyLine(
            locations=route_coords,
            color=color,
            weight=4,
            opacity=0.8,
            tooltip=f"Vehicle {v_idx}"
        ).add_to(m)


    # Save and open
    map_filename = "vrp_routes_map.html"
    m.save(map_filename)
    os.system(f"start {map_filename}")

def main():

    # Network initialization and pre-processing helper class
    network_builder = NetworkBuilder(number_of_nodes=20, num_vehicles=4)


    data = network_builder.create_data_model()

    solver = VRPSolver(data=data)

    print(data)

    solver.solve()


if __name__ == '__main__':
    main()