import os
import osmnx as ox
import numpy as np
import networkx as nx
import traceback
from networkx import MultiGraph
from multiprocessing import Process, shared_memory

import time

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

    def load_map_data(self) -> MultiGraph:
        return ox.io.load_graphml(filepath="./data/Quezon.graphml")

    def get_sample_nodes(self, graph) -> Any:
        number_of_nodes = 20
        nodes = list(graph.nodes)

        return np.random.choice(nodes,number_of_nodes, replace=False)
    


    def select_connected_nodes(self, graph):
        """
        Randomly samples nodes directly from a connected graph.
        Assumes the input graph is already a single connected component.
        """
        all_nodes = list(graph.nodes)
        
        if len(all_nodes) < self.number_of_nodes:
            print(f"Warning: Graph has only {len(all_nodes)} nodes; adjusting sample size.")
            self.number_of_nodes = len(all_nodes)

        return np.random.choice(all_nodes, self.number_of_nodes, replace=False).tolist()



    def _worker(self, graph, nodes, node_idx_map, chunk_indices, shm_name, shape, dtype):

        # Access shared memory from parent process
        shm = shared_memory.SharedMemory(name=shm_name)

        # Get the distance_matrix from shared memory region
        distance_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # Calculate for each chunk
        for i in chunk_indices:

            # Source variable contains the actual node object
            source = nodes[i]

            # single source dijkstra path length calculates the shortest path length from a given node to all nodes in the input graph
            lengths = nx.single_source_dijkstra_path_length(graph, source, weight="length")

            # Loop through all the lengths and assemble in shared memory region
            for target, dist in lengths.items():
                if target in node_idx_map:
                    j = node_idx_map[target]
                    distance_matrix[i, j] = dist


    def calculate_distance_matrix_parallel(self, graph, num_workers=7):
        print("Starting to calculate for distance matrix using multiprocessing...")
   
        start_calculate_matirx_time = time.time()

        nodes = self.sampled_nodes

        # node_idx_map is for mapping specific nodes in actual distance caclculation 
        node_idx_map = {node: i for i, node in enumerate(nodes)}

        # Define the dimension of the distance_matrix
        shape = (len(nodes), len(nodes))

        # Define the type of the actual distance values
        dtype = np.float64

        # Initialize shared momery instance for interprocess communication technique
        shm = shared_memory.SharedMemory(
            create=True,
            size=int(np.prod(shape)) * np.dtype(dtype).itemsize
        )

        # Put distance_matrix object to shared memory region and initialy fill with zeros
        distance_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        distance_matrix.fill(0.0)

        # Calculate chunk size basd on number of workers and nodes
        chunk_size = int(np.ceil(len(nodes) / num_workers))

        # Create an array of indices and delegate to processes
        chunks = [list(range(i, min(i + chunk_size, len(nodes)))) for i in range(0, len(nodes), chunk_size)]

        # Creates a Process object to calculate per chunk
        processes = []
        for chunk in chunks:
            p = Process(target=self._worker, args=(graph, nodes, node_idx_map, chunk, shm.name, shape, dtype))
            p.start()
            processes.append(p)

        # Waits for all processes to finish
        for p in processes:
            p.join()

        # Copy and return the distance_matrix from shared memory region
        final_matrix = distance_matrix.copy()

        # Close and clear shared memory instance
        shm.close()
        shm.unlink()
    
        end_calculate_matirx_time = time.time()
        print(f'Distance matrix calculated in {end_calculate_matirx_time - start_calculate_matirx_time} seconds')
        return final_matrix


    # Serial version that uses shortest_path_length function call for every cell
    def calculate_distance_matrix(self, graph) -> List[List[Union[int,float]]]:
        
        # Get sampled nodes from graph
        # nodes = self.get_sample_nodes(graph=graph)
        # nodes = self.get_connected_nodes(graph=graph)
        print('Starting to calculate for distance matrix....')
        start_calculate_matirx_time = time.time()
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

        end_calculate_matirx_time = time.time()
        print(f'Distance matrix calculated in {end_calculate_matirx_time - start_calculate_matirx_time} seconds')
        
        return distance_matrix
    
    def create_data_model(self) -> List[List[Union[int,float]]]:

        # 1. Load graph from data file
        print("Loading Map data....")
        graph:MultiGraph = self.load_map_data()

        print("Starting to select and sample connected nodes....")
        start_select_nodes_time = time.time()
        self.sampled_nodes = self.select_connected_nodes(graph=graph)
        end_select_nodes_time = time.time()
        print(f'Found connected nodes to sample in {end_select_nodes_time - start_select_nodes_time} seconds')

    
        self.node_positions = {
            node: (graph.nodes[node]['y'], graph.nodes[node]['x']) for node in self.sampled_nodes
        }
        
        
        # 2. Init data structure
        data = {}

        # 3. Calculate distance for each sampled node
        # data['distance_matrix'] = self.calculate_distance_matrix(graph)
        data['distance_matrix'] = self.calculate_distance_matrix_parallel(graph)

        # 4. Init related data
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
        print('Starting to solve for solution....')
        start_solve_solution_time = time.time()
        transit_callback_index = self.routing.RegisterTransitCallback(self.distance_callback)

        # Define cost of each arc
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


        # Add Distance constraint.
        dimension_name = "Distance"
        self.routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            14485,  # vehicle maximum travel distance
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

        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 200

        search_parameters.log_search = True

        # Solve the problem
        try:
            solution = self.routing.SolveWithParameters(search_parameters)
            if solution:
                
                end_solve_solution_time = time.time()
                print(f'Solution found in {end_solve_solution_time - start_solve_solution_time} seconds')
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
    colors = [
    'red', 'blue', 'green', 'orange', 'purple', 'pink', 'lightblue', 'lightgreen', 'beige', 'yellow',
    'lightred', 'cadetblue', 'cyan', 'lime', 'magenta', 'gold', 'aqua', 'lavender', 'coral', 'turquoise',
    'salmon', 'plum', 'khaki', 'tomato', 'deepskyblue', 'mediumseagreen', 'springgreen', 'dodgerblue', 'orchid', 'greenyellow',
    'lightcoral', 'mediumturquoise', 'peachpuff', 'skyblue', 'hotpink', 'wheat', 'chartreuse', 'powderblue', 'mediumorchid', 'darkorange',
    'lightpink', 'palegreen', 'lightsalmon', 'lightcyan', 'mediumvioletred', 'aquamarine', 'darkturquoise', 'moccasin', 'mistyrose', 'lemonchiffon'
    ]


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
    network_builder = NetworkBuilder(number_of_nodes=10000, num_vehicles=10000)


    data = network_builder.create_data_model()

    solver = VRPSolver(data=data)

    print(data)

    solver.solve()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    
    runtime = end - start
    print(f"The total runtime of the program: {runtime} seconds")