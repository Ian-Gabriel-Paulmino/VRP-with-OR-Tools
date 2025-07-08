from core_modules import NetworkBuilder, VRPSolver

import time

def main():

    # Network initialization and pre-processing helper class
    network_builder = NetworkBuilder(number_of_nodes=2000, num_vehicles=2000, vehicle_capacity=100, demand_sampling='power_law')


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