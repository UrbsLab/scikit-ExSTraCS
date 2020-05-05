
class ClassifierSet:
    def __init__(self):
        self.popSet = []  # List of classifiers/rules
        self.matchSet = []  # List of references to rules in population that match
        self.correctSet = []  # List of references to rules in population that both match and specify correct phenotype
        self.microPopSize = 0