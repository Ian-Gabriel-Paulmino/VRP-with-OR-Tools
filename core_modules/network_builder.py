import networkx as nx
import osmnx as ox
from networkx import MultiGraph

import numpy as np
import time
import random

from multiprocessing import Process, shared_memory

from typing import Any, List, Union


class NetworkBuilder:

    # Needs max demand, number of instances, save to directory in problem instances
    # Exact nodes to solve
    def __init__(self, number_of_nodes: int, num_vehicles: int, vehicle_capacity: int, max_demand: int, demand_sampling: str, customer_position: str, depot_position: str):
        self.number_of_nodes = number_of_nodes
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.demand_sampling = demand_sampling
        self.customer_position = customer_position
        self.depot_position = depot_position
        self.max_demand = max_demand

        self.sampled_nodes = []  # store selected node IDs
        self.node_positions = {}  # store {node_id: (lat, lon)}

    def load_map_data(self) -> MultiGraph:
        return ox.io.load_graphml(filepath="./data/Quezon.graphml")

    def get_sample_nodes(self, graph) -> Any:
        number_of_nodes = 20
        nodes = list(graph.nodes)

        return np.random.choice(nodes,number_of_nodes, replace=False)
    
    def get_depot_position(self, graph):
        
        if self.depot_position == 'random':
            all_nodes = list(graph.nodes)

            return np.random.choice(all_nodes, replace=False)
        
        if self.depot_position == 'center':
            return min(nx.center(graph))
    

    # ================================================================================
    # Customer location sampling
    # ================================================================================
    def select_connected_nodes(self, graph) -> List[Any]:
        """
        Randomly samples nodes directly from a connected graph.
        Assumes the input graph is already a single connected component.
        """
        all_nodes = list(graph.nodes)
        
        depot = self.get_depot_position(graph)

        if len(all_nodes) < self.number_of_nodes:
            print(f"Warning: Graph has only {len(all_nodes)} nodes; adjusting sample size.")
            self.number_of_nodes = len(all_nodes)

        return [depot] + np.random.choice(all_nodes, self.number_of_nodes-1, replace=False).tolist()

    def single_cluster_sampling(self,graph):

        all_nodes = list(graph.nodes)

        depot = self.get_depot_position(graph)
        cluster_start = np.random.choice(all_nodes, replace=False)

        bfs_tree = nx.bfs_tree(graph, source=cluster_start)

        bfs_nodes = list(bfs_tree.nodes)

        return [depot] + bfs_nodes[:self.number_of_nodes]
    


    


    def random_cluster_sampling(self, graph, min_cluster_size=20, max_cluster_size=50):
        all_nodes = list(graph.nodes)
        depot = self.get_depot_position(graph)

        remaining = self.number_of_nodes
        cluster_sizes = []

        # Randomly break up the total number of nodes
        while remaining > 0:
            if remaining < min_cluster_size:
                size = remaining  # Just take the rest if it's smaller than min cluster size
            else:
                size = random.randint(min_cluster_size, min(max_cluster_size, remaining))
            cluster_sizes.append(size)
            remaining -= size

        final_sampled_nodes = set()
        used_seeds = set()

        for size in cluster_sizes:
            # Select a new unique seed node
            seed = None
            attempts = 0
            max_attempts = 100

            while attempts < max_attempts:
                candidate = np.random.choice(all_nodes)
                if candidate not in used_seeds:
                    seed = candidate
                    used_seeds.add(seed)
                    break
                attempts += 1

            if seed is None:
                # fallback: skip this cluster if no valid seed found
                continue

            # Perform BFS
            bfs_tree = nx.bfs_tree(graph, source=seed)
            bfs_nodes = list(bfs_tree.nodes)
            cluster_sample = [n for n in bfs_nodes if n not in final_sampled_nodes][:size]
            final_sampled_nodes.update(cluster_sample)

            if len(final_sampled_nodes) >= self.number_of_nodes:
                break

        # Final adjustment in case we slightly oversampled
        final_sampled_nodes = list(final_sampled_nodes)
        if len(final_sampled_nodes) > self.number_of_nodes:
            final_sampled_nodes = final_sampled_nodes[:self.number_of_nodes]

        result = [depot] + final_sampled_nodes
        print(f"Total nodes sampled is: {len(result)}")
        return result
        



    # ================================================================================
    # Distance Matrix solver w/ multiprocessing
    # ================================================================================
    def _worker(self, graph, nodes, node_idx_map, chunk_indices, shm_name, shape, dtype) -> None:

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


    def calculate_distance_matrix_parallel(self, graph, num_workers=7) -> List[List[Union[int,float]]]:
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
    

    # ================================================================================
    # Customer demand sampling
    # ================================================================================

    # def bounded_powerlaw_sampling(self, xmin, xmax, alpha, size=1):
    #     values = np.arange(xmin, xmax + 1).astype(float)
    #     weights = values ** -alpha
    #     probabilities = weights / weights.sum()
    #     return np.random.choice(values, size=size, p=probabilities)

    def bounded_powerlaw_sampling(self, xmin, xmax, alpha, size=1):
        # Inverse transform sampling for discrete power law
        r = np.random.uniform(0, 1, size)
        
        values = np.arange(xmin, xmax + 1)
        weights = values.astype(float) ** -alpha
        norm = np.sum(weights)
        cumulative = np.cumsum(weights / norm)
        
        # Match random values to the discrete cumulative distribution
        return values[np.searchsorted(cumulative, r)]

    def get_customer_demand(self):
        """
        Returns customer demand using the defined sampling method
        """
        if self.demand_sampling == 'random':
            # Random Sampling
            return [0] + np.random.choice(np.arange(1, 10), size=len(self.sampled_nodes) - 1).tolist()
        elif self.demand_sampling == 'power_law':
            # Power Law Sampling
            return [0] + self.bounded_powerlaw_sampling(1, self.max_demand, alpha=3, size=len(self.sampled_nodes) - 1).tolist()
    

    def get_customer_positions(self, graph):

        if self.customer_position == 'random':
            return self.select_connected_nodes(graph)
        elif self.customer_position == 'clustered':
            return self.single_cluster_sampling(graph)
        elif self.customer_position == 'random-clustered':
            return self.random_cluster_sampling(graph)


    def create_data_model(self) -> List[List[Union[int,float]]]:

        # 1. Load graph from data file
        print("Loading Map data....")
        graph:MultiGraph = self.load_map_data()

        print("Starting to select and sample connected nodes....")
        start_select_nodes_time = time.time()

        self.sampled_nodes = self.get_customer_positions(graph)
        end_select_nodes_time = time.time()
        print(f'Found connected nodes to sample in {end_select_nodes_time - start_select_nodes_time} seconds')

        # Saving node_positions (lat,lon) for visualization in folium
        self.node_positions = {
            node: (graph.nodes[node]['y'], graph.nodes[node]['x']) for node in self.sampled_nodes
        }
        
        
        # 2. Init data structure
        data = {}

        # 3. Calculate distance for each sampled node
        data['distance_matrix'] = self.calculate_distance_matrix_parallel(graph)

        # 4. Init related data
        data["num_vehicles"] = self.num_vehicles
        data["depot"] = 0
        data['graph'] = graph
        data['node_ids'] = self.sampled_nodes
        data['positions'] = self.node_positions
        data['demand_sampling'] = self.demand_sampling

        data['customer_position'] = self.customer_position
        data['depot_position'] = self.depot_position

        # Capacity constraint
        data['vehicle_capacities'] = [self.vehicle_capacity] * self.num_vehicles

        # Customer demand
        data['demands'] = self.get_customer_demand()

        return data