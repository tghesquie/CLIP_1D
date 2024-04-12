class Dumper:
    def __init__(self):
        # Initialize lists to store the results
        self.data = {
            "imposed_displacement": [],
            "force": [],  
            "displacement": [],
            "cohesive_damage": [],
            "bulk_damage": [],
            "lagrange": [],
            "stress": [],
            "jump": [],
            "functional": [],
            "cohesive_stress": [],
            "strain": [],
            "cohesive_dissip_expected": [],
            "bulk_dissip_expected": [],
            "total_dissip_expected":[],
            "cohesive_dissip_actual": [],
            "bulk_dissip_actual": [],
            "total_dissip_actual" : [],
        }
        
    def store(self, key, value):
        if key in self.data:
            self.data[key].append(value)
        else:
            print(f"Warning: Key '{key}' not recognized.")