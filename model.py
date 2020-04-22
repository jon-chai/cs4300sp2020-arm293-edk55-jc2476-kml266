import sys
import pickle
import ast
import pandas as pd
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

class IngredientIndex:
    ''' Class representing an inverted index search tool'''

    def __init__(self):

        self.recipes={}
        self.inv_idx={}

    def process(self):

        with open('RAW_recipes.csv', encoding='utf-8') as f:

            raw_recipes = pd.read_csv(f)

            for idx, row in raw_recipes.iterrows():

                if type(row['name']) is not str or row['name'] is '':
                    continue

                row_id = row['id']
                self.recipes[row_id] = {}
                self.recipes[row_id]['name'] = row['name']
                self.recipes[row_id]['description'] = row['description']
                self.recipes[row_id]['ingredients'] = ast.literal_eval(row['ingredients'])
                self.recipes[row_id]['steps'] = ast.literal_eval(row['steps'])
                self.recipes[row_id]['nutrition'] = ast.literal_eval(row['nutrition'])
                self.recipes[row_id]['tags'] = ast.literal_eval(row['tags'])

                self.recipes[row_id]['features'] = set()

                for ingredient in self.recipes[row_id]['ingredients']:
                    self.recipes[row_id]['features'].add(lemmatizer.lemmatize(ingredient))

                    single_ingredient_list = ingredient.split()

                    if len(single_ingredient_list) != 1:

                        for single_ingredient in single_ingredient_list:
                            self.recipes[row_id]['features'].add(lemmatizer.lemmatize(single_ingredient))

                for ingredient in self.recipes[row_id]['name'].split():
                    self.recipes[row_id]['features'].add(lemmatizer.lemmatize(ingredient))

                for ingredient in self.recipes[row_id]['tags']:
                    self.recipes[row_id]['features'].add(lemmatizer.lemmatize(ingredient))

                for ingredient in self.recipes[row_id]['features']:
                    if ingredient not in self.inv_idx:
                        self.inv_idx[ingredient] = set()
                    self.inv_idx[ingredient].add(row_id)

    def get_id(self, item):
        return self.inv_idx[lemmatizer.lemmatize(item)]

    def __getitem__(self, item):
        recipe_ids = self.get_id(item)
        recipe_names = set()
        for recipe in recipe_ids:
            recipe_names.add(self.recipes[recipe]['name'])
        return recipe_names

    # Returns set of recipes such that every ingredient in ingredient_list is included in each recipe
    def intersect(self, ingredient_list):

        result = -1

        for ingredient in ingredient_list:
            if result == -1:
                result = self.get_id(ingredient)
            else:
                result = result.intersection(self.get_id(ingredient))

        if result == -1:
            result = set()

        return result

    # Returns set of recipes such that at least one ingredient in ingredient_list is included in each recipe
    def union(self, ingredient_list):

        result = set()

        for ingredient in ingredient_list:
            result = result.union(self.get_id(ingredient))

        return result

    # Returns set of recipes such that no ingredient in ingredient_list is included in each recipe
    def exclude(self, ingredient_list):

        bad_recipes = self.union(ingredient_list)

        result = set()

        for recipe in self.recipes:
            if recipe not in bad_recipes:
                result.add(recipe)

        return result

    # Returns set of recipes such that every ingredient in good_ingredient_list is included in each recipe
    # and every ingredient in bad_ingredient_list is not included in each recipe
    def search(self, good_ingredient_list, bad_ingredient_list):

        return {self.recipes[recipe_id]['name'] for recipe_id in
                self.intersect(good_ingredient_list).intersection(self.exclude(bad_ingredient_list))}

if __name__ == '__main__':
    Pickler = IngredientIndex()
    Pickler.process()
    with open('model.pkl', 'wb') as f:
        pickle.dump(Pickler, f)
