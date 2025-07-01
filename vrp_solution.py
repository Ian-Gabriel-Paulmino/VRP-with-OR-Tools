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

    def load_map_data(self) -> MultiDiGraph:
        return ox.io.load_graphml(filepath="./data/Quezon.graphml")

    def get_sample_nodes(self, graph) -> Any:
        number_of_nodes = 20
        nodes = list(graph.nodes)

        return np.random.choice(nodes,number_of_nodes, replace=False)

    def calculate_distance_matrix(self, graph) -> List[List[Union[int,float]]]:
        
        # Get sampled nodes from graph
        nodes = self.get_sample_nodes(graph=graph)

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

        # 2. Convert to projected graph
        projected_graph:MultiDiGraph = ox.project_graph(graph)
        
        # 3. Init data structure
        data = {}

        # 4. Calculate distance for each sampled node
        data['distance_matrix'] = self.calculate_distance_matrix(projected_graph)

        # 5. Init related data
        data["num_vehicles"] = self.num_vehicles
        data["depot"] = 0

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
            30000,  # vehicle maximum travel distance
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
            else:
                print("No solution found (but no crash).")
        except Exception:
            traceback.print_exc()
            



def main():

    # Network initialization and pre-processing helper class
    network_builder = NetworkBuilder(number_of_nodes=20, num_vehicles=4)


    data = network_builder.create_data_model()

    solver = VRPSolver(data=data)

    print(data)

    solver.solve()


if __name__ == '__main__':
    main()