import copy
from typing import Dict, List

class Package:
    def __init__(self, id: int, x: float, y: float, weight: float, priority: int):
        self.id = id
        self.destination = (x, y)  # (x, y) coordinates
        self.weight = weight
        self.priority = priority  # 1 (highest) to 5 (lowest)

    def __repr__(self):
        return f"Package({self.id}, dest={self.destination}, weight={self.weight}kg, priority={self.priority})"


class Vehicle:
    def __init__(self, id: int, capacity: float):
        self.id = id
        self.capacity = capacity

    def __repr__(self):
        return f"Vehicle({self.id}, capacity={self.capacity}kg)"


class Solution:
    def __init__(self, assignments: Dict[int, List[int]]):
        """assignments: Dictionary mapping vehicle IDs to lists of package IDs"""
        # عنا الديكشيناري يمثل الانتيجر رقم المركبة و الليست بتمثل الطرود الي جواه 
        self.assignments = assignments

    def copy(self):
        """Create a deep copy of the solution"""
        
        return Solution(copy.deepcopy(self.assignments))

    def __repr__(self):
        return f"Solution({self.assignments})"
