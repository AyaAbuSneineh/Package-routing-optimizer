import time
from typing import Callable, Tuple, List, Dict, Any

from data_structures import Package, Vehicle, Solution
from utils import (visualize_solution, print_solution_details,plot_temperature_history,plot_costItt_history)
from simulated_annealing import simulated_annealing
from genetic_algorithm import genetic_algorithm, NUMBER_OF_GENERATIONS


def run_algorithm(algorithm_name: str, packages: List[Package], vehicles: List[Vehicle],
                    params: Dict[str, Any]) -> Tuple[Solution, List[float], Dict[str, Any]]:
    """Run the selected algorithm with the given parameters"""
    start_time = time.time()
    # Choose algorithm
    if algorithm_name == "simulated_annealing":
        cooling_rate = params.get("cooling_rate")
        solution, cost_history, stats = simulated_annealing(packages, vehicles, cooling_rate)
    else :
        population_size = params.get("population_size")
        mutation_rate = params.get("mutation_rate")
        solution, cost_history, stats = genetic_algorithm(packages, vehicles, population_size, mutation_rate, NUMBER_OF_GENERATIONS)

    # Add elapsed time to stats
    elapsed_time = time.time() - start_time
    stats["elapsed_time"] = elapsed_time

    return solution, cost_history, stats

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def read_input_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    Vehicles = []
    num_vehicles = int(lines[0])
    if num_vehicles <= 0:
        print(f"The provided number of vehicles ({num_vehicles}) is not valid . It must be greater than zero.")
        return None, None, 0, 0

    max_capacity = 0
    valid_vehicle = 0
    j = 1 # to take id for vehicle 
    for i in range(1, num_vehicles + 1):
        Vehicle_data = lines[i].split()
        if len(Vehicle_data) != 1:
            print(f"Invalid vehicle data format at line {i + 1}.")
            continue
        if not is_float(Vehicle_data[0]):
            print(f"Invalid vehicle capacity format at line {i + 1}. Must be a numeric value.")
            continue

        capacity = float(Vehicle_data[0])

        if capacity <= 0:
            print(f"Invalid vehicle capacity at line {i + 1}. Capacity must be positive.")
            continue
        
        if capacity > max_capacity: 
            max_capacity = capacity
        
        
        Vehicle_id = j
        j += 1
        Vehicles.append(Vehicle(Vehicle_id, capacity))
        valid_vehicle += 1
    if valid_vehicle == 0:
        print("No valid vehicles found. Exiting the program.")
        return None, None, 0, 0 
    packages = []
    num_packages = 0
    for line in lines[num_vehicles + 1:]:
        package_data = line.split()
        if len(package_data) != 4:
            print(f"Invalid package data format: {line}. Expected: x y weight priority.")
            continue

        if not (package_data[0].isdigit() and package_data[1].isdigit() and is_float(package_data[2]) and package_data[3].isdigit()):
            print(f"Line {lines.index(line) + 1} contains invalid (non-numeric) values. Skipping.")
            continue
        
        x = int(package_data[0])
        y = int(package_data[1])
        weight = float(package_data[2])
        priority = int(package_data[3])
        if x < 0 or x > 100:
            print(f"Invalid X coordinate for package at line {lines.index(line) + 1}. X must be between 0 and 100.")
            continue
        if y < 0 or y > 100:
            print(f"Invalid Y coordinate for package at line {lines.index(line) + 1}. Y must be between 0 and 100.")
            continue
        if priority < 1 or priority > 5:
            print(f"Invalid priority for package at line {lines.index(line) + 1}. Priority must be between 1 and 5.")
            continue
        
        if weight <= 0 or weight > max_capacity:
            print(f"Invalid weight for package at line {lines.index(line) + 1}. Weight must be between 0 and {max_capacity}.")
            continue
        num_packages += 1
        packages.append(Package(num_packages,x, y, weight, priority))
    #print(num_packages)
    return Vehicles, packages, num_packages, valid_vehicle


def main():
    """Main function to run the package delivery optimization"""
    filename = "input.txt"
    vehicles, packages, num_packages, valid_vehicles = read_input_file(filename)
    if valid_vehicles <=0 :
        print("No vehicle available. Exiting program.")
        return
    if num_packages <= 0 :
        print("No packages available. Exiting program")
        return
    else:
        print(f"Valid vehicles ({valid_vehicles}):")
        print(f"Number of valid packages: {num_packages}")
    """else:
        print(f"Valid vehicles ({valid_vehicles}):")
        for vehicle in vehicles:
            print(vehicle)
        print(f"\nNumber of valid packages: {num_packages}")
        for package in packages:
                print(package)"""

    print("Package Delivery Optimization")
    while True : 
        # Algorithm selection
        print(f"Select optimization algorithm:")
        print("1. Simulated Annealing")
        print("2. Genetic Algorithm")
        algorithm_choice = int(input("Enter your choice (1 or 2): "))
        if algorithm_choice == 1:
            algorithm_name = "simulated_annealing"
            cooling_rate = float(input("Enter cooling rate (0.90-0.99): "))
            if  cooling_rate < 0.9 or cooling_rate > 0.99:
                while True :
                    cooling_rate = float(input("Enter cooling rate (0.90-0.99): "))
                    if cooling_rate >= 0.9 and cooling_rate <= 0.99:
                        break
            params = {"cooling_rate": cooling_rate}
            break

        elif algorithm_choice == 2:
            algorithm_name = "genetic_algorithm"
            population_size = int(input(f"Enter population size (50-100) : "))
            if population_size < 50 or population_size > 100 :
                while True :
                    population_size = int(input(f"Enter population size (50-100) : "))
                    if population_size >= 50 and  population_size <= 100 :
                        break
            mutation_rate = float(input(f"Enter mutation rate (0.01-0.1): "))
            if mutation_rate < 0.01 or mutation_rate > 0.1 :
                while True :
                    mutation_rate = float(input(f"Enter mutation rate (0.01-0.1): "))
                    if mutation_rate >= 0.01 and mutation_rate <= 0.1 :
                        break
            params = {"population_size": population_size, "mutation_rate": mutation_rate}
            break

        else:
            print("Invalid choice! Please select 1 or 2.")

    assigned_package_ids = set()
    # Run optimization
    if algorithm_name == "simulated_annealing" :
        print(f"\nRunning {algorithm_name}...")
        solution, cost_history, stats = run_algorithm(algorithm_name, packages, vehicles, params)
    # Print results
        for pkg_list in solution.assignments.values():
            assigned_package_ids.update(pkg_list)

        assigned_count = len(assigned_package_ids)
        unassigned_count = num_packages - assigned_count
        print(f"Total number of  Packages: {num_packages}")
        print(f"Assigned Packages: {assigned_count}")
        print(f"Unassigned Packages: {unassigned_count}")
        print(f"\n{algorithm_name} completed in {stats['elapsed_time']:.2f} seconds")
        print_solution_details(solution, packages, vehicles, stats, algorithm_name)

    # Visualize results
        print("\nVisualizing solution...")
        visualize_solution(solution, packages, vehicles)
        #plot_cost_history(cost_history, algorithm_name)
        plot_temperature_history(stats['temperature_history'])
        plot_costItt_history(cost_history)

    else:
        print(f"\nRunning {algorithm_name}...")
        solution, cost_history, stats = run_algorithm(algorithm_name, packages, vehicles, params)
    # Print results
        for pkg_list in solution.assignments.values():
            assigned_package_ids.update(pkg_list)

        assigned_count = len(assigned_package_ids)
        unassigned_count = num_packages - assigned_count

        print(f"Total number of  Packages: {num_packages}")
        print(f"Assigned Packages: {assigned_count}")
        print(f"Unassigned Packages: {unassigned_count}")
        print(f"\n{algorithm_name} completed in {stats['elapsed_time']:.2f} seconds")
        print_solution_details(solution, packages, vehicles, stats, algorithm_name)

    # Visualize results
        print("\nVisualizing solution...")
        visualize_solution(solution, packages, vehicles)
        #plot_cost_history(cost_history, algorithm_name)

    print("\nOptimization complete!") 

if __name__ == "__main__":
    main()