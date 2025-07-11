from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from utilities.visualizations import visualize_solution

from typing import Union, List

import time
import traceback


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
    
    # Add Capacity constraint.
    def demand_callback(self,from_index) -> int:
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex
        from_node = self.manager.IndexToNode(from_index)
        return self.data["demands"][from_node]
    




    # =================================
    # Solution printers
    # =================================

    def print_solution_default(self, solution) -> None:
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

    def print_solution_capacitated(self, solution) -> None:
        """Prints solution on console."""
        print(f"Objective: {solution.ObjectiveValue()}")
        total_distance = 0
        total_load = 0
        for vehicle_id in range(self.data["num_vehicles"]):
            if not self.routing.IsVehicleUsed(solution, vehicle_id):
                continue
            index = self.routing.Start(vehicle_id)
            plan_output = f"Route for vehicle {vehicle_id}:\n"
            route_distance = 0
            route_load = 0
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                route_load += self.data["demands"][node_index]
                plan_output += f" {node_index} Load({route_load}) -> "
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            plan_output += f" {self.manager.IndexToNode(index)} Load({route_load})\n"
            plan_output += f"Distance of the route: {route_distance}m\n"
            plan_output += f"Load of the route: {route_load}\n"
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
        print(f"Total distance of all routes: {total_distance}m")
        print(f"Total load of all routes: {total_load}")


    # Extracts solution data to be used in saving results
    def extract_solution_data(self, solution) -> dict:
        total_distance = 0
        total_load = 0
        vehicle_routes = []

        for vehicle_id in range(self.data["num_vehicles"]):
            if not self.routing.IsVehicleUsed(solution, vehicle_id):
                continue

            index = self.routing.Start(vehicle_id)
            route_distance = 0
            route_load = 0
            route = []

            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                route_load += self.data["demands"][node_index]
                route.append({
                    "node": node_index,
                    "load": route_load
                })
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )

            # End depot
            route.append({
                "node": self.manager.IndexToNode(index),
                "load": route_load
            })

            total_distance += route_distance
            total_load += route_load

            vehicle_routes.append({
                "vehicle_id": vehicle_id,
                "route": route,
                "distance": route_distance,
                "load": route_load
            })

        return {
            "objective_value": solution.ObjectiveValue(),
            "total_distance": total_distance,
            "total_load": total_load,
            "vehicle_routes": vehicle_routes
        }







    def solve(self) -> None:
        print('Starting to solve for solution....')
        start_solve_solution_time = time.time()
        transit_callback_index = self.routing.RegisterTransitCallback(self.distance_callback)

        # Define cost of each arc
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


        # Add Distance constraint.
        # Last-mile delivery vehicles run up to 6-9 miles (10 miles used here, converted to meters)
        # dimension_name = "Distance"
        # self.routing.AddDimension(
        #     transit_callback_index,
        #     0,  # no slack
        #     30000,  # vehicle maximum travel distance
        #     True,  # start cumul to zero
        #     dimension_name,
        # )
        # distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
        # distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Add capacity constraint
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(self.demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.data["vehicle_capacities"],  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
        )
        search_parameters.time_limit.seconds = 900

        search_parameters.log_search = True

        # Solve the problem
        try:
            solution = self.routing.SolveWithParameters(search_parameters)
            if solution:
                
                end_solve_solution_time = time.time()
                print(f'Solution found in {end_solve_solution_time - start_solve_solution_time} seconds')
                self.print_solution_capacitated(solution)
                routes = self.get_routes(solution)
                # visualize_solution(self.data, routes)
                return solution
            else:
                print("No solution found (but no crash).")
                return False
        except Exception:
            traceback.print_exc()

    # Helper function to iterate through solution and get routes for visualization
    def get_routes(self, solution) -> List[List[int]]:

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