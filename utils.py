import random
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

from data_structures import Package, Vehicle, Solution

# Constants
SHOP_LOCATION = (0, 0)
GRID_SIZE = 100  # The grid is 100km x 100km للخريطة
PRIORITY_WEIGHT = 20  # Weight for priority cost in the overall cost function


def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def generate_initial_solution(packages: List[Package], vehicles: List[Vehicle]) -> Solution:
    """Generate a random initial solution that respects vehicle capacity constraints"""

    # Create a list of package IDs to assign
    #unassigned_packages = [p.id for p in packages] # just only store the ids of packeges 
    #random.shuffle(unassigned_packages)   # random packeges id
    sorted_packages = sorted(packages, key=lambda p: p.priority)
    unassigned_packages = [p.id for p in sorted_packages]
     

    assignments = dict() # Create empty assignment for each vehicle
    remaining_capacity = dict() # Dictionary to track remaining capacity of each vehicle
    for v in vehicles:
        assignments[v.id] = []
        remaining_capacity[v.id] = v.capacity

    # Assign packages to vehicles while respecting capacity
    for pkg_id in unassigned_packages:
        for p in packages:
            if p.id == pkg_id:
                pkg = p
                break

        # Find vehicles that can accommodate this package
        valid_vehicles = [v.id for v in vehicles if remaining_capacity[v.id] >= pkg.weight]

        if valid_vehicles:
            # Randomly select a vehicle that can accommodate this package
            vehicle_id = random.choice(valid_vehicles)
            assignments[vehicle_id].append(pkg_id)
            remaining_capacity[vehicle_id] -= pkg.weight

    # For each vehicle, randomly shuffle the order of package delivery
    for vehicle_id in assignments: # هون برضو رجعنا رتبنا الطرود بشكل عشوائي لكل مركبة
        random.shuffle(assignments[vehicle_id])

    return Solution(assignments)

#Calculate the total cost of a solution
def calculate_cost(solution: Solution, packages: List[Package], vehicles: List[Vehicle]) -> Tuple[float, float, float]:
    
    # Dictionary to look up packages by ID
    package_dict = {p.id: p for p in packages} # هون بقدر أوصل للباكيج من خلال الاي دي 

    total_distance = 0  # Total distance for all vehicles
    priority_cost = 0   # Total penalty due to priority violations

    for vehicle_id, package_ids in solution.assignments.items():
        if not package_ids:  # Skip empty routes اذا كانت المركبة فش فيها ولا طرد
            continue

        # Start at the shop
        current_location = SHOP_LOCATION #(0,0)
        distance_traveled = 0 
        delivered_packages = []

        # Calculate route distance
        for i, pkg_id in enumerate(package_ids):
            package = package_dict[pkg_id]
            distance_to_package = euclidean_distance(current_location, package.destination)
            package_shop = euclidean_distance(SHOP_LOCATION, package.destination)

 # Check if any previously delivered lower-priority package was delivered earlier
            for delivered_id in delivered_packages:
                delivered = package_dict[delivered_id]
                delivered_shop = euclidean_distance(SHOP_LOCATION, delivered.destination)
                if delivered.priority > package.priority and (distance_traveled > package_shop or package_shop < delivered_shop):
                    delay_ratio = (distance_traveled - package_shop) / package_shop
                    penalty = PRIORITY_WEIGHT * delay_ratio * (6 - package.priority)
                    priority_cost += penalty
                    break # Stop after the first violation

            delivered_packages.append(pkg_id)
            current_location = package.destination
            distance_traveled += distance_to_package
# Add vehicle's total travel distance including return to shop
        total_distance += distance_traveled 
        total_distance += euclidean_distance(current_location, SHOP_LOCATION)
    total_cost = total_distance + priority_cost
    return (total_cost, total_distance, priority_cost)


def is_capacity_valid(solution: Solution, packages: List[Package], vehicles: List[Vehicle]) -> bool:
    """Check if the solution respects all vehicle capacity constraints"""
    package_dict = {p.id: p for p in packages}
    vehicle_dict = {v.id: v for v in vehicles}

    for vehicle_id, package_ids in solution.assignments.items():
        vehicle_capacity = vehicle_dict[vehicle_id].capacity
        total_weight = sum(package_dict[pkg_id].weight for pkg_id in package_ids)

        if total_weight > vehicle_capacity:
            return False

    return True


def visualize_solution(solution: Solution, packages: List[Package], vehicles: List[Vehicle]):
    """Visualize the solution on a 2D grid"""
    package_dict = {p.id: p for p in packages}

    plt.figure(figsize=(12, 10))

    # Plot the shop (depot)
    plt.plot(SHOP_LOCATION[0], SHOP_LOCATION[1], 'ks', markersize=15, label='Shop')

    # Plot each vehicle's route with a different color
    colors = plt.cm.tab10(np.linspace(0, 1, len(vehicles)))

    for i, vehicle_id in enumerate(solution.assignments):
        if not solution.assignments[vehicle_id]:
            continue  # Skip empty routes

        route_x = [SHOP_LOCATION[0]]
        route_y = [SHOP_LOCATION[1]]

        # Add each package destination in order
        for pkg_id in solution.assignments[vehicle_id]:
            package = package_dict[pkg_id]
            route_x.append(package.destination[0])
            route_y.append(package.destination[1])

            # Plot package with color based on priority
            priority_color = plt.cm.RdYlGn_r(package.priority / 5)  # Red for high priority, yellow-green for low
            plt.plot(package.destination[0], package.destination[1], 'o',
                     color=priority_color, markersize=8)

            # Add package label
            plt.text(package.destination[0] + 1, package.destination[1] + 1,
                     f"P{pkg_id}(w:{package.weight:.1f},pr:{package.priority})")

        # Return to shop
        route_x.append(SHOP_LOCATION[0])
        route_y.append(SHOP_LOCATION[1])

        # Plot the route
        plt.plot(route_x, route_y, '-', color=colors[i],
                 label=f"Vehicle {vehicle_id} (cap: {vehicles[i].capacity:.1f}kg)")

    # Set plot limits and labels
    #plt.xlim(-5, GRID_SIZE + 5)
    #plt.ylim(-5, GRID_SIZE + 5)
    # Dynamic scaling based on package locations
    all_x = [SHOP_LOCATION[0]]
    all_y = [SHOP_LOCATION[1]]
    for pkg in packages:
        all_x.append(pkg.destination[0])
        all_y.append(pkg.destination[1])

    margin = 5
    plt.xlim(min(all_x) - margin, max(all_x) + margin)
    plt.ylim(min(all_y) - margin, max(all_y) + margin)

    plt.xlabel('X coordinate (km)')
    plt.ylabel('Y coordinate (km)')
    plt.title('Package Delivery Routes')
    plt.grid(True)
    plt.legend(loc='upper right')
    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_temperature_history(temperature_history: list):
    """Plot the temperature drop over iterations"""
    plt.figure(figsize=(10, 6))
    plt.plot(temperature_history, color='orange')

    plt.xlabel("Iterations")
    plt.ylabel("Temperature")
    plt.title("Temperature Drop During Simulated Annealing")
    plt.grid(True)
    plt.show()

def plot_costItt_history(cost_history):
    """Plot the Best Cost drop over iterations"""
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, color='green')

    plt.xlabel("Iterations")
    plt.ylabel("Best Cost")
    plt.title("Progress of Best Cost in Simulated Annealing")
    plt.grid(True)
    plt.show()


def print_solution_details(solution: Solution, packages: List[Package], vehicles: List[Vehicle], stats: dict,
                           algorithm_name: str):
    """Print detailed information about the solution"""
    package_dict = {p.id: p for p in packages}

    print(f"\n=== {algorithm_name} SOLUTION DETAILS ===")
    print(f"Total Cost: {stats['best_cost']:.2f}")
    print(f"Distance Cost: {stats['best_distance']:.2f} km")
    print(f"Priority Cost: {stats['best_priority_cost']:.2f}")
    print("\nRoute Details:")

    for vehicle_id, package_ids in solution.assignments.items():
        if not package_ids:
            print(f"Vehicle {vehicle_id}: Not used")
            continue

        vehicle = next(v for v in vehicles if v.id == vehicle_id)
        total_weight = sum(package_dict[pkg_id].weight for pkg_id in package_ids)
        utilization = (total_weight / vehicle.capacity) * 100

        print(f"\nVehicle {vehicle_id} (Capacity: {vehicle.capacity:.1f}kg, Used: {total_weight:.1f}kg, {utilization:.1f}%)")
        print("Delivery Route:")

        current_location = SHOP_LOCATION
        total_distance = 0

        for i, pkg_id in enumerate(package_ids):
            package = package_dict[pkg_id]
            distance = euclidean_distance(current_location, package.destination)
            total_distance += distance

            print(f"  Stop {i + 1}: Package {pkg_id} (Weight: {package.weight:.1f}kg, "
                  f"Priority: {package.priority}, Distance: {distance:.2f}km)")

            current_location = package.destination

        # Return to shop
        distance_back = euclidean_distance(current_location, SHOP_LOCATION)
        total_distance += distance_back

        print(f"  Return to Shop (Distance: {distance_back:.2f}km)")
        print(f"  Total Route Distance: {total_distance:.2f}km")

    print("\nAlgorithm Stats:")
    keys=["iterations","accepted_moves","rejected_moves","temperature_drops","best_cost","best_distance","best_priority_cost", "generations",
        "population_size","mutation_rate"]
    for key in keys:
        if key in stats:
            value = stats[key]
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")