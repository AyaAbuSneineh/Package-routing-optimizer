import random
from typing import List, Tuple, Callable

from data_structures import Package, Vehicle, Solution
from utils import calculate_cost, generate_initial_solution

# Constants specific to Genetic Algorithm
NUMBER_OF_GENERATIONS = 500  # Fixed value per requirements


def create_initial_population(packages: List[Package], vehicles: List[Vehicle], size: int) -> List[Solution]:
    """Create an initial population of valid solutions for the genetic algorithm"""
    population = []
    for _ in range(size):
        solution = generate_initial_solution(packages, vehicles)
        population.append(solution)
    return population


def select_parents(population: List[Solution], fitness_scores: List[float]) -> Tuple[Solution, Solution]:
    """Select two parents using tournament selection"""
    # Tournament selection
    tournament_size = 5

    # First parent
    candidates1 = random.sample(list(range(len(population))), tournament_size)
    best_candidate1 = min(candidates1, key=lambda idx: fitness_scores[idx])
    parent1 = population[best_candidate1]

    # Second parent (ensure it's different from the first)
    # Create a list of available indices excluding the first chosen candidate
    available_indices = [i for i in range(len(population)) if i != best_candidate1]
    candidates2 = random.sample(available_indices, min(tournament_size, len(available_indices)))
    best_candidate2 = min(candidates2, key=lambda idx: fitness_scores[idx])
    parent2 = population[best_candidate2]

    return parent1, parent2


def crossover(parent1: Solution, parent2: Solution, packages: List[Package], vehicles: List[Vehicle]) -> Tuple[
    Solution, Solution]:
    """
    Perform crossover between two parent solutions to create two children.
    This improved implementation prevents duplicate package assignments.
    """
    # Initialize empty children
    child1 = Solution({v.id: [] for v in vehicles})
    child2 = Solution({v.id: [] for v in vehicles})

    # Package dictionary for weight lookup
    package_dict = {p.id: p for p in packages}

    # Get all package IDs that need to be assigned
    all_packages = set()
    for vehicle_id in parent1.assignments:
        all_packages.update(parent1.assignments[vehicle_id])

    # Track which packages have been assigned to each child
    assigned_to_child1 = set()
    assigned_to_child2 = set()

    # Track remaining capacity for each vehicle
    remaining_capacity1 = {v.id: v.capacity for v in vehicles}
    remaining_capacity2 = {v.id: v.capacity for v in vehicles}

    # Choose a crossover point
    crossover_point = random.randint(1, len(vehicles) - 1) if len(vehicles) > 1 else 0

    # Perform crossover for each vehicle
    for i, vehicle in enumerate(vehicles):
        vehicle_id = vehicle.id

        if i < crossover_point:
            # Child1 gets packages from parent1, Child2 from parent2
            source1 = parent1.assignments[vehicle_id]
            source2 = parent2.assignments[vehicle_id]
        else:
            # Child1 gets packages from parent2, Child2 from parent1
            source1 = parent2.assignments[vehicle_id]
            source2 = parent1.assignments[vehicle_id]

        # Assign packages to child1
        for pkg_id in source1:
            # Skip if already assigned or exceeds capacity
            if pkg_id in assigned_to_child1:
                continue

            child1.assignments[vehicle_id].append(pkg_id)
            assigned_to_child1.add(pkg_id)
            remaining_capacity1[vehicle_id] -= package_dict[pkg_id].weight

        # Assign packages to child2
        for pkg_id in source2:
            # Skip if already assigned or exceeds capacity
            if pkg_id in assigned_to_child2:
                continue

            child2.assignments[vehicle_id].append(pkg_id)
            assigned_to_child2.add(pkg_id)
            remaining_capacity2[vehicle_id] -= package_dict[pkg_id].weight

    # Assign remaining unassigned packages to vehicles with sufficient capacity
    unassigned_for_child1 = all_packages - assigned_to_child1
    unassigned_for_child2 = all_packages - assigned_to_child2

    # Assign remaining packages for child1
    for pkg_id in unassigned_for_child1:
        # Find vehicles with sufficient capacity
        valid_vehicles = []
        for v in vehicles:
            if remaining_capacity1[v.id] >= package_dict[pkg_id].weight:
                valid_vehicles.append(v.id)

        if valid_vehicles:
            # Randomly choose a valid vehicle
            vehicle_id = random.choice(valid_vehicles)
            child1.assignments[vehicle_id].append(pkg_id)
            remaining_capacity1[vehicle_id] -= package_dict[pkg_id].weight

    # Assign remaining packages for child2
    for pkg_id in unassigned_for_child2:
        # Find vehicles with sufficient capacity
        valid_vehicles = []
        for v in vehicles:
            if remaining_capacity2[v.id] >= package_dict[pkg_id].weight:
                valid_vehicles.append(v.id)

        if valid_vehicles:
            # Randomly choose a valid vehicle
            vehicle_id = random.choice(valid_vehicles)
            child2.assignments[vehicle_id].append(pkg_id)
            remaining_capacity2[vehicle_id] -= package_dict[pkg_id].weight

    return child1, child2

def mutate(solution: Solution, packages: List[Package], vehicles: List[Vehicle], mutation_rate: float) -> Solution:
    """Apply mutation to a solution based on the mutation rate"""
    if random.random() > mutation_rate:
        return solution  # No mutation

    # Choose a mutation type
    mutation_type = random.choice(['swap_packages', 'change_order' , 'Move'])

    # Apply the selected mutation
    if mutation_type == 'swap_packages':
        # Find two vehicles with packages
        non_empty_vehicles = [v_id for v_id, pkgs in solution.assignments.items() if pkgs]
        if len(non_empty_vehicles) >= 2:
            v1, v2 = random.sample(non_empty_vehicles, 2)

            if solution.assignments[v1] and solution.assignments[v2]:
                # Try to swap packages
                p1_idx = random.randrange(len(solution.assignments[v1]))
                p2_idx = random.randrange(len(solution.assignments[v2]))

                # Check capacity constraints
                package_dict = {p.id: p for p in packages}
                vehicle_dict = {v.id: v for v in vehicles}

                p1_id = solution.assignments[v1][p1_idx]
                p2_id = solution.assignments[v2][p2_idx]

                # Calculate weights
                v1_weight = sum(package_dict[pid].weight for pid in solution.assignments[v1])
                v2_weight = sum(package_dict[pid].weight for pid in solution.assignments[v2])

                # Calculate weight differences
                weight_diff1 = package_dict[p2_id].weight - package_dict[p1_id].weight
                weight_diff2 = package_dict[p1_id].weight - package_dict[p2_id].weight

                # Check if swap is valid
                if (v1_weight + weight_diff1 <= vehicle_dict[v1].capacity and
                        v2_weight + weight_diff2 <= vehicle_dict[v2].capacity):
                    # Perform the swap
                    solution.assignments[v1][p1_idx] = p2_id
                    solution.assignments[v2][p2_idx] = p1_id

    elif mutation_type == 'change_order':
        # Choose a vehicle and reorder its packages
        non_empty_vehicles = [v_id for v_id, pkgs in solution.assignments.items() if len(pkgs) >= 2]
        if non_empty_vehicles:
            v_id = random.choice(non_empty_vehicles)
            # Just shuffle the route
            random.shuffle(solution.assignments[v_id])
    elif mutation_type == 'Move':
        # Find two vehicles with packages
        non_empty_vehicles = [v_id for v_id, pkgs in solution.assignments.items() if pkgs]
        if len(non_empty_vehicles) >= 2:
            v1, v2 = random.sample(non_empty_vehicles, 2)

            if solution.assignments[v1] and solution.assignments[v2]:
                # Try to swap packages
                p1_idx = random.randrange(len(solution.assignments[v1]))

                # Check capacity constraints
                package_dict = {p.id: p for p in packages}
                vehicle_dict = {v.id: v for v in vehicles}

                p1_id = solution.assignments[v1][p1_idx]

                # Calculate weights
                v2_weight = sum(package_dict[pid].weight for pid in solution.assignments[v2])

                # Check if swap is valid
                if (v2_weight + package_dict[p1_id].weight <= vehicle_dict[v2].capacity):
                    # Perform the swap
                    solution.assignments[v1].pop(p1_idx)
                    solution.assignments[v2].append(p1_id)
    return solution


def genetic_algorithm(packages: List[Package], vehicles: List[Vehicle],population_size: int,mutation_rate: float,num_generations: int = NUMBER_OF_GENERATIONS,) -> Tuple[Solution, list, dict]:
    """
    Implement a genetic algorithm for package delivery optimization
    Returns the best solution found, cost history, and stats
    """
    # Create initial population
    population = create_initial_population(packages, vehicles, population_size)

    # Evaluate the initial population
    fitness_scores = []
    for solution in population:
        cost, distance, priority = calculate_cost(solution, packages, vehicles)
        fitness_scores.append(cost)  # Lower cost = higher fitness

    # Find the best solution in the initial population
    best_idx = fitness_scores.index(min(fitness_scores))
    best_solution = population[best_idx].copy()
    best_cost, best_distance, best_priority = calculate_cost(best_solution, packages, vehicles)

    # For tracking progress
    cost_history = [best_cost]
    stats = {
        'generations': num_generations,
        'population_size': population_size,
        'mutation_rate': mutation_rate,
        'best_cost': best_cost,
        'best_distance': best_distance,
        'best_priority_cost': best_priority
    }

    # Main genetic algorithm loop
    for generation in range(num_generations):
        # Create next generation
        new_population = []
        new_fitness_scores = []

        # top: Keep the best solution(s)
        top_count = max(1, int(population_size * 0.05))  # Keep top 5%
        top_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[0:top_count]

        for idx in top_indices:
            new_population.append(population[idx].copy())
            new_fitness_scores.append(fitness_scores[idx])

        # Create the rest of the population through selection, crossover, and mutation
        while len(new_population) < population_size:
            # Select parents
            parent1, parent2 = select_parents(population, fitness_scores)

            # Crossover
            child1, child2 = crossover(parent1, parent2, packages, vehicles)

            # Mutation
            child1 = mutate(child1, packages, vehicles, mutation_rate)
            child2 = mutate(child2, packages, vehicles, mutation_rate)

            # Add to new population
            for child in [child1, child2]:
                if len(new_population) < population_size:
                    new_population.append(child)
                    cost, _, _ = calculate_cost(child, packages, vehicles)
                    new_fitness_scores.append(cost)

        # Update population and fitness scores
        population = new_population
        fitness_scores = new_fitness_scores

        # Update best solution
        current_best_idx = fitness_scores.index(min(fitness_scores))
        current_best_solution = population[current_best_idx]
        current_best_cost, current_distance, current_priority = calculate_cost(current_best_solution, packages,
                                                                               vehicles)

        if current_best_cost < best_cost:
            best_solution = current_best_solution.copy()
            best_cost = current_best_cost
            best_distance = current_distance
            best_priority = current_priority
            stats['best_cost'] = best_cost
            stats['best_distance'] = best_distance
            stats['best_priority_cost'] = best_priority

        # Record progress
        cost_history.append(best_cost)

        # Report progress
        """if (generation + 1) % 20 == 0:
            print(f"Generation {(generation + 1)}/{num_generations}, Best Cost: {best_cost:.2f}")
"""
    return best_solution, cost_history, stats