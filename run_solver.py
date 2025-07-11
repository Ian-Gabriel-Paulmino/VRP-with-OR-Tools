from core_modules import NetworkBuilder, VRPSolver

import time
import os
import numpy as np
import json

from datetime import datetime


def save_instance(instance_id, data):
    import json
    base_path = f"cvrp_experiments/instances/{instance_id}"
    os.makedirs(base_path, exist_ok=True)

    np.save(f"{base_path}/distance_matrix.npy", data['distance_matrix'])

    metadata = {
        'sampled_nodes': [int(n) for n in data['node_ids']],
        'positions': {str(int(k)): (float(v[0]), float(v[1])) for k, v in data['positions'].items()},
        'vehicle_capacity': int(data['vehicle_capacities'][0]),
        'demand_sampling': str(data['demand_sampling']),
        'customer_position': str(data['customer_position']),
        'depot_position': str(data['depot_position']),
        'num_vehicles': int(data['num_vehicles'])
    }

    with open(f"{base_path}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)



def run_solver_on_instance(instance_id, new_vehicle_capacity, demand_sampling, run_id, network_builder):
    base_path = f"cvrp_experiments/instances/{instance_id}"
    solution_path = f"cvrp_experiments/solutions/{instance_id}"
    os.makedirs(solution_path, exist_ok=True)

    with open(f"{base_path}/metadata.json") as f:
        meta = json.load(f)
    distance_matrix = np.load(f"{base_path}/distance_matrix.npy")

    # Prepare demand with selected sampling method
    network_builder.demand_sampling = demand_sampling
    demands = network_builder.get_customer_demand()

    # Construct data
    data = {
        'distance_matrix': distance_matrix,
        'num_vehicles': meta['num_vehicles'],
        'vehicle_capacities': [new_vehicle_capacity] * meta['num_vehicles'],
        'depot': 0,
        'demands': demands,
        'node_ids': meta['sampled_nodes'],
        'positions': meta['positions']
    }

    solver = VRPSolver(data)

    # Time the solution
    start_time = time.time()
    solution = solver.solve()
    end_time = time.time()
    solve_duration = end_time - start_time

    if solution:
        result_data = solver.extract_solution_data(solution)
        result_data["run_parameters"] = {
            "vehicle_capacity": new_vehicle_capacity,
            "demand_sampling": demand_sampling,
            "run_id": run_id,
            "time_taken_seconds": solve_duration
        }

        result_data["demands"] = [int(d) for d in demands]

        # Save solution + demands in one file
        with open(f"{solution_path}/run_{run_id:02d}_vehicle{new_vehicle_capacity}_{demand_sampling}.json", 'w') as f:
            json.dump(result_data, f, indent=4)

        # Save solution data
        with open(f"{solution_path}/run_{run_id:02d}_vehicle{new_vehicle_capacity}_{demand_sampling}.json", 'w') as f:
            json.dump(result_data, f, indent=4)
    else:
        print(f"No solution found for run {run_id}")


def run_multiple_solver_configurations_on_one_instance(instance_id, network_builder, number_of_runs, parameter_variations):
    print("Creating a single instance...")
    data = network_builder.create_data_model()
    save_instance(instance_id, data)

    print(f"Running {number_of_runs} solver runs on the same instance...\n")
    for run_id in range(number_of_runs):
        params = parameter_variations[run_id]
        run_solver_on_instance(
            instance_id=instance_id,
            new_vehicle_capacity=params["vehicle_capacity"],
            demand_sampling=params["demand_sampling"],
            run_id=run_id + 1,
            network_builder=network_builder
        )



def main():

    # Multiple instance configurations
    instance_variations = [
        {"number_of_nodes": 500, "num_vehicles": 500, "customer_position": "clustered", "depot_position": "random"},
        {"number_of_nodes": 1000, "num_vehicles": 1000 , "customer_position": "random-clustered", "depot_position": "random"},
        {"number_of_nodes": 1500, "num_vehicles": 1500,  "customer_position": "random-clustered", "depot_position": "random"},
        {"number_of_nodes": 2000, "num_vehicles": 2000, "customer_position": "random", "depot_position": "random"},
    ]

    # Experiment configurations for each instance (solver runs)
    parameter_variations = [
        {"vehicle_capacity": 50, "demand_sampling": "power_law"},
        {"vehicle_capacity": 75, "demand_sampling": "random"},
        {"vehicle_capacity": 100, "demand_sampling": "power_law"},
        {"vehicle_capacity": 200, "demand_sampling": "power_law"}
    ]

    for instance_config in instance_variations:
        # Add shared constants
        instance_config_full = {
            **instance_config,
            "vehicle_capacity": 0,  # will be overwritten per run
            "max_demand": 20,
            "demand_sampling": "",  # will be overwritten per run
        }

        # Initialize builder for this instance config
        network_builder = NetworkBuilder(**instance_config_full)

        # Generate dynamic and descriptive instance ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        instance_id = (
            f"{network_builder.number_of_nodes}_customers_"
            f"{network_builder.customer_position}_customer_pos_"
            f"{network_builder.depot_position}_depot_pos_{timestamp}"
        )

        # Run multiple solver configs on this instance
        run_multiple_solver_configurations_on_one_instance(
            instance_id=instance_id,
            network_builder=network_builder,
            number_of_runs=len(parameter_variations),
            parameter_variations=parameter_variations
        )

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    
    runtime = end - start
    print(f"The total runtime of the program: {runtime} seconds")