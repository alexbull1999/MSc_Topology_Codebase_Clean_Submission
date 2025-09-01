import json
import random
from itertools import combinations

class ToyEntailmentGenerator:
    def __init__(self):
        self.entity_properties = {
            "animals": {
                "entities": ["dogs", "cats", "birds", "fish", "elephants", "tigers"],
                "valid_properties": ["brown", "large", "fast", "friendly", "wild", "intelligent", "strong", "beautiful"],
                "contradictory_properties": ["wild", "domestic"]
            },
            "vehicles": {
                "entities": ["cars", "trucks", "motorcycles", "bicycles", "buses"],
                "valid_properties": ["red", "fast", "expensive", "large", "useful", "modern", "reliable"],
                "contradictory_properties": ["expensive", "cheap"]
            },
            "plants": {
                "entities": ["roses", "oaks", "tulips", "pines", "daisies"],
                "valid_properties": ["beautiful", "large", "colorful", "natural", "tall", "fragrant"],
                "contradictory_properties": ["edible", "inedible"]
            },
            "objects": {
                "entities": ["books", "chairs", "tables", "computers", "phones"],
                "valid_properties": ["useful", "expensive", "modern", "large", "reliable", "portable"],
                "contradictory_properties": ["heavy", "light"]
            }
        }

        self.properties = [
            "brown", "friendly", "loud", "fast", "small", "large",
            "beautiful", "strong", "intelligent", "colourful", "wild",
            "domestic", "expensive", "useful", "natural"
        ]

        self.hierarchy = {
            "animal": ["mammal", "bird", "fish"],
            "mammal": ["dog", "cat"],
            "vehicle": ["car", "truck", "motorcycle"],
            "plant": ["tree", "flower"],
            "tree": ["oak"],
            "flower": ["rose"]
        }

        self.attribute_hierarchy = {
            "coloured": ["red", "blue", "green", "yellow"],
            "measurable": ["tiny", "small", "medium", "large", "huge"]
        }

    def generate_logical_pairs(self, n_pairs=1000):
        """Generate logical entailment pairs"""
        pairs = set()

        #Type 1: Universal to Existential
        universal_existential = []
        for category_name, category_data in self.entity_properties.items():
            entities = category_data["entities"]
            properties = category_data["valid_properties"]

            for entity in entities:
                for property in properties:
                    premise = f"All {entity} are {property}"
                    hypothesis = f"Some {entity} are {property}"
                    universal_existential.append((premise, hypothesis, "entailment"))
                    universal_existential.append((hypothesis, premise, "neutral"))

                    hypothesis_contra = f"No {entity} are {property}"
                    universal_existential.append((premise, hypothesis_contra, "contradiction"))


        pairs.update(universal_existential)

        #Type 2: Hierarchical relationships
        hierarchical = []
        for parent, children in self.hierarchy.items():
            if parent == "animal":
                for child in children:
                    premise = f"This object is a {child}"
                    hypothesis = f"This object is an {parent}"
                    hierarchical.append((premise, hypothesis, "entailment"))
                    hierarchical.append((hypothesis, premise, "neutral"))
            else:
                for child in children:
                    premise = f"This object is a {child}"
                    hypothesis = f"This object is a {parent}"
                    hierarchical.append((premise, hypothesis, "entailment"))
                    hierarchical.append((hypothesis, premise, "neutral"))


        for parent, children in self.attribute_hierarchy.items():
            for child in children:
                premise = f"This object is {child}"
                hypothesis = f"This object is {parent}"
                hierarchical.append((premise, hypothesis, "entailment"))
                hierarchical.append((hypothesis, premise, "neutral"))

        pairs.update(hierarchical)

        #Property combinations (AND/OR logic)
        combination_pairs = []
        for category_name, category_data in self.entity_properties.items():
            entities = category_data["entities"]
            properties = category_data["valid_properties"]

            selected_categories = random.sample(entities, min(3, len(entities)))
            selected_properties = random.sample(properties, min(3, len(properties)))

            for category in selected_categories:
                for prop1, prop2 in combinations(selected_properties, 2):
                    # Conjunction: X is A and B -> X is A (entailment)
                    premise = f"This {category.rstrip('s')} is {prop1} and {prop2}"
                    hypothesis = f"This {category.rstrip('s')} is {prop1}"
                    combination_pairs.append((premise, hypothesis, "entailment"))
                    combination_pairs.append((hypothesis, premise, "neutral"))

        pairs.update(combination_pairs)

        # Contradiction pairs
        contradiction_pairs = []
        for category_name, category_data in self.entity_properties.items():
            entities = category_data["entities"]
            properties = category_data["contradictory_properties"]
            selected_categories = random.sample(entities, min(3, len(entities)))

            for category in selected_categories:
                for prop1, prop2 in combinations(properties, 2):
                    premise = f"This {category.rstrip('s')} is {prop1}"
                    hypothesis = f"This {category.rstrip('s')} is {prop2}"
                    contradiction_pairs.append((premise, hypothesis, "contradiction"))

        pairs.update(contradiction_pairs)

        pairs_list = list(pairs)
        random.shuffle(pairs_list)

        return pairs_list[:n_pairs]

    def analyze_dataset_balance(self, pairs):
        """Analyze the balance of the generated dataset"""
        labels = [pair[2] for pair in pairs]

        print(f"Dataset Analysis:")
        print(f"Total pairs: {len(pairs)}")
        print(f"Entailment: {labels.count('entailment')} ({labels.count('entailment') / len(pairs) * 100:.1f}%)")
        print(f"Neutral: {labels.count('neutral')} ({labels.count('neutral') / len(pairs) * 100:.1f}%)")
        print(
            f"Contradiction: {labels.count('contradiction')} ({labels.count('contradiction') / len(pairs) * 100:.1f}%)")

        # Check for duplicates
        premises = [pair[0] for pair in pairs]
        hypotheses = [pair[1] for pair in pairs]
        print(f"Unique premises: {len(set(premises))}")
        print(f"Unique hypotheses: {len(set(hypotheses))}")

        return {
            'total': len(pairs),
            'entailment': labels.count('entailment'),
            'neutral': labels.count('neutral'),
            'contradiction': labels.count('contradiction'),
            'unique_premises': len(set(premises)),
            'unique_hypotheses': len(set(hypotheses))
        }

    def create_balanced_dataset(self, target_size=300):
        """Create a perfectly balanced dataset"""
        # Generate all possible pairs
        all_pairs = self.generate_logical_pairs(2000)  # Generate more than needed

        # Separate by label
        entailment_pairs = [p for p in all_pairs if p[2] == "entailment"]
        neutral_pairs = [p for p in all_pairs if p[2] == "neutral"]
        contradiction_pairs = [p for p in all_pairs if p[2] == "contradiction"]

        # Take equal numbers of each
        per_label = target_size // 3

        balanced_pairs = (
                random.sample(entailment_pairs, min(per_label, len(entailment_pairs))) +
                random.sample(neutral_pairs, min(per_label, len(neutral_pairs))) +
                random.sample(contradiction_pairs, min(per_label, len(contradiction_pairs)))
        )

        random.shuffle(balanced_pairs)
        return balanced_pairs

    def save_dataset(self, pairs, filename):
        with open(f"data/toy/{filename}", 'w') as f:
            json.dump(pairs, f, indent=2)

        analysis = self.analyze_dataset_balance(pairs)
        with open(f"data/toy/{filename.replace('.json', '_analysis.json')}", 'w') as f:
            json.dump(analysis, f, indent=2)




generator = ToyEntailmentGenerator()
datasets = {
    "small": generator.create_balanced_dataset(300),
    "medium": generator.create_balanced_dataset(600),
    "large": generator.generate_logical_pairs(1000)
}

for name, pairs in datasets.items():
    print(f"\n{name.upper()} DATASET:")
    generator.save_dataset(pairs, f"{name}_logical_pairs.json")

