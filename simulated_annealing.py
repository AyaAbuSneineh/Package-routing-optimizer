import random
import math
from typing import List, Tuple, Callable

from data_structures import Package, Vehicle, Solution
from utils import calculate_cost, generate_initial_solution

# Constants specific to Simulated Annealing
INITIAL_TEMPERATURE = 1000
MIN_TEMPERATURE = 1
ITERATIONS_PER_TEMP = 100
temperature_history = [INITIAL_TEMPERATURE]


def generate_neighbor(current_solution: Solution, packages: List[Package], vehicles: List[Vehicle]) -> Solution:
    #Generate a neighbor solution by applying a random move.
    #Ensures that the neighbor respects vehicle capacity constraints.
    
    package_dict = {p.id: p for p in packages}
    vehicle_dict = {v.id: v for v in vehicles}

    # Try to generate a valid neighbor
    max_attempts = 100
    for _ in range(max_attempts):
        # Create a deep copy of the current solution
        neighbor = current_solution.copy()

        # Choose a move type
        #swap : تبديل طرود بين مركبتين بشرط ما يتجاوزوا السعة
        #reorder : إعادة ترتيب الطرود داخل مركبة واحدة
        # move : نقل طرد من مركبة ل مركبة ثانية 
        move_type = random.choice(['swap', 'reorder','move'])

        # Get non-empty vehicle routes
        non_empty_vehicles = [v_id for v_id, pkg_ids in neighbor.assignments.items() if pkg_ids]

        if not non_empty_vehicles:
            continue  # No packages assigned, can't make a move

        if move_type == 'swap' and len(non_empty_vehicles) >= 2:
            # Swap packages between vehicles
            vehicle1, vehicle2 = random.sample(non_empty_vehicles, 2)

# pick 1 package from each vehicle and swap them, but before swamping them make sure that their weight can fit in the new vehicle
            if neighbor.assignments[vehicle1] and neighbor.assignments[vehicle2]:
                pkg1_idx = random.randrange(len(neighbor.assignments[vehicle1]))
                pkg2_idx = random.randrange(len(neighbor.assignments[vehicle2]))

                pkg1_id = neighbor.assignments[vehicle1][pkg1_idx]
                pkg2_id = neighbor.assignments[vehicle2][pkg2_idx]

                # Check if swap would violate capacity
                vehicle1_capacity = vehicle_dict[vehicle1].capacity
                vehicle2_capacity = vehicle_dict[vehicle2].capacity

                # Calculate current weights
                vehicle1_weight = sum(package_dict[pid].weight for pid in neighbor.assignments[vehicle1])
                vehicle2_weight = sum(package_dict[pid].weight for pid in neighbor.assignments[vehicle2])

                # Calculate weight change
                weight_change1 = package_dict[pkg2_id].weight - package_dict[pkg1_id].weight
                weight_change2 = package_dict[pkg1_id].weight - package_dict[pkg2_id].weight

                # Check if swap is valid
                if (vehicle1_weight + weight_change1 <= vehicle1_capacity and
                    vehicle2_weight + weight_change2 <= vehicle2_capacity):
                    # Perform the swap
                    neighbor.assignments[vehicle1][pkg1_idx] = pkg2_id
                    neighbor.assignments[vehicle2][pkg2_idx] = pkg1_id
                    return neighbor

        elif move_type == 'reorder' and non_empty_vehicles:
            # Reorder packages within a vehicle
            vehicle_id = random.choice(non_empty_vehicles)
            if len(neighbor.assignments[vehicle_id]) >= 2:
                idx1, idx2 = random.sample(range(len(neighbor.assignments[vehicle_id])), 2)
                # Swap the positions of two packages in the route
                (neighbor.assignments[vehicle_id][idx1],
                 neighbor.assignments[vehicle_id][idx2]) = (neighbor.assignments[vehicle_id][idx2],
                                                            neighbor.assignments[vehicle_id][idx1])
                return neighbor
        elif move_type == 'move' and len(non_empty_vehicles) >= 2:
            # Move one package from one vehicle to another
            from_vehicle, to_vehicle = random.sample(non_empty_vehicles, 2)

            if neighbor.assignments[from_vehicle]:
                pkg_idx = random.randrange(len(neighbor.assignments[from_vehicle]))
                pkg_id = neighbor.assignments[from_vehicle][pkg_idx]

                pkg_weight = package_dict[pkg_id].weight
                to_vehicle_weight = sum(package_dict[pid].weight for pid in neighbor.assignments[to_vehicle])
                to_vehicle_capacity = vehicle_dict[to_vehicle].capacity

                # Check if the receiving vehicle has enough capacity
                if to_vehicle_weight + pkg_weight <= to_vehicle_capacity:
                    # Move the package
                    neighbor.assignments[from_vehicle].pop(pkg_idx)
                    neighbor.assignments[to_vehicle].append(pkg_id)
                    return neighbor

    # If we couldn't generate a valid neighbor, return a copy of the current solution
    return current_solution.copy()


def simulated_annealing(packages: List[Package], vehicles: List[Vehicle],cooling_rate: float,) -> Tuple[Solution, list, dict]:
    """
    Implement the simulated annealing algorithm for package delivery optimization
    Returns the best solution found, cost history, and stats
    """
    # Generate initial solution
    current_solution = generate_initial_solution(packages, vehicles)
    best_solution = current_solution.copy()

    # Calculate initial costs
    current_cost, current_distance, current_priority = calculate_cost(current_solution, packages, vehicles)
    best_cost = current_cost

    # Initialize temperature
    temperature = INITIAL_TEMPERATURE

    # For tracking progress
    cost_history = [best_cost]
    iteration = 0
    stats = {
        'iterations': 0,
        'accepted_moves': 0,
        'rejected_moves': 0,
        'temperature_drops': 0,
        'best_cost': best_cost,
        'best_distance': current_distance,
        'temperature_history' : 0 ,
        'cost_historyitt' : 0 ,
        'best_priority_cost': current_priority
    }

    # Main simulated annealing loop
    while temperature > MIN_TEMPERATURE:
        for _ in range(ITERATIONS_PER_TEMP):
            iteration += 1
            temperature_history.append(temperature)
            cost_history.append(best_cost)

            # Generate a neighbor
            neighbor = generate_neighbor(current_solution, packages, vehicles)

            # Calculate the new cost
            neighbor_cost, neighbor_distance, neighbor_priority = calculate_cost(neighbor, packages, vehicles)

            # Calculate cost difference
            delta_cost = neighbor_cost - current_cost

            # Accept or reject the neighbor
            if delta_cost < 0:  # Better solution - always accept
                current_solution = neighbor
                current_cost = neighbor_cost
                stats['accepted_moves'] += 1

                # Update best if this is better
                if current_cost < best_cost:
                    best_solution = neighbor.copy()
                    best_cost = current_cost
                    stats['best_cost'] = best_cost
                    stats['best_distance'] = neighbor_distance
                    stats['best_priority_cost'] = neighbor_priority
            else:
                # Worse solution - accept with probability
                acceptance_probability = math.exp(-delta_cost / temperature)
                if random.random() < acceptance_probability:
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    stats['accepted_moves'] += 1
                else:
                    stats['rejected_moves'] += 1

        # Cool down the temperature
        temperature *= cooling_rate
        stats['temperature_drops'] += 1
        #cost_history.append(best_cost)
        
        # Print progress
        """if stats['temperature_drops'] % 10 == 0:
            print(f"Temperature: {temperature:.2f}, Best Cost: {best_cost:.2f}")
            """

    stats['iterations'] = iteration
    stats['temperature_history'] = temperature_history

    return best_solution, cost_history, stats