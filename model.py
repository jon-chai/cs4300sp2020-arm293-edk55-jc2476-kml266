import sys
import pickle
import ast
import pandas as pd
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import gensim.downloader as api
import numpy as np

def process():

    recipes = {}
    inv_idx = {}
    flavors = ['sweet', 'salty', 'savory', 'sour', 'spicy', 'bitter', 'citrus', 'fruity', 'rich']
    stop_words = set(stopwords.words('english'))
    wv = api.load('glove-twitter-25')
    
    with open('RAW_interactions.csv', encoding='utf-8') as f:
        
        raw_interactions = pd.read_csv(f)
        valid_recipes = {}
        
        for idx, row in raw_interactions.iterrows():
            r_id = row['recipe_id']
            if r_id not in valid_recipes:
                valid_recipes[r_id] = []
            valid_recipes[r_id].append(row['rating'])
        
    good_valid_recipes = {}
    extra = [3] * 5

    for r_id, rating_list in valid_recipes.items():
        mean = np.mean(rating_list + extra)
        if len(rating_list) != 6 and mean >= 4:
            good_valid_recipes[r_id] = mean
        
        
    with open('RAW_recipes.csv', encoding='utf-8') as f:

        raw_recipes = pd.read_csv(f)

        for idx, row in raw_recipes.iterrows():

            if type(row['name']) is not str or row['name'] is '' or row['id'] not in good_valid_recipes:
                continue

            row_id = row['id']
            recipes[row_id] = {}
            
            recipes[row_id]['name'] = row['name']
            recipes[row_id]['id']=row['id']
            recipes[row_id]['description'] = row['description']
            recipes[row_id]['ingredients'] = ast.literal_eval(row['ingredients'])
            recipes[row_id]['tags'] = ast.literal_eval(row['tags'])
            recipes[row_id]['flavors'] = {}
            recipes[row_id]['rating'] = good_valid_recipes[row_id]

            recipes[row_id]['features'] = set()

            for ingredient in recipes[row_id]['ingredients']:
                recipes[row_id]['features'].add(lemmatizer.lemmatize(ingredient))

                single_ingredient_list = ingredient.split()

                if len(single_ingredient_list) != 1:

                    for single_ingredient in single_ingredient_list:
                        recipes[row_id]['features'].add(lemmatizer.lemmatize(single_ingredient))

            for ingredient in recipes[row_id]['name'].split():
                recipes[row_id]['features'].add(lemmatizer.lemmatize(ingredient))
            
            

            for ingredient in recipes[row_id]['tags']:
                recipes[row_id]['features'].add(lemmatizer.lemmatize(ingredient))

            for ingredient in recipes[row_id]['features']:
                if ingredient not in inv_idx:
                    inv_idx[ingredient] = set()
                inv_idx[ingredient].add(row_id)
                     
                    
            good_ingredients = [lemmatizer.lemmatize(ingredient) for ingredient in recipes[row_id]['name'].split() if ingredient not in stop_words]
            
            for flavor in flavors:
                sim = 0
                tot = .01
                for ingredient in good_ingredients:
                    try:
                        sim += wv.similarity(ingredient, flavor)
                        tot += 1
                    except KeyError:
                        continue
                        
                recipes[row_id]['flavors'][flavor] = round(sim / tot, 2)    
                    
    return recipes, inv_idx

class IngredientIndex:
    ''' Class representing an inverted index search tool'''

    def __init__(self, recipes, inv_idx):

        self.recipes = recipes
        self.inv_idx = inv_idx
        self.flavors = set(item for item in ['sweet', 'salty', 'savory', 'sour', 'spicy', 'bitter', 'citrus', 'fruity', 'rich'])

    def get_id(self, item):
        item = lemmatizer.lemmatize(item)
        if item in self.inv_idx:
            return self.inv_idx[lemmatizer.lemmatize(item)]
        return set()

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
    def search(self, good_ingredient_list, bad_ingredient_list, flavor_list):

#         good_recipes =  [(recipe_id,self.recipes[recipe_id]) for recipe_id in
#                 self.intersect(good_ingredient_list).intersection(self.exclude(bad_ingredient_list))]
        
        good_recipes =  [(recipe_id, 0) for recipe_id in self.intersect(good_ingredient_list).intersection(self.exclude(bad_ingredient_list))]
        new_recipes = []
        flavor_list = [flavor for flavor in flavor_list if flavor in self.flavors]
        print(flavor_list)
        n_flavors = len(flavor_list)
        
        if n_flavors != 0:
            for recipe_id, score in good_recipes:
                for flavor in flavor_list:
                    score += self.recipes[recipe_id]['flavors'][flavor]
                new_recipes.append((recipe_id, score))

            good_recipes = sorted(new_recipes, key=lambda x: x[1], reverse=True)
            
        return [(x[0], self.recipes[x[0]]) for x in good_recipes]
           
                    
if __name__ == '__main__':
    recipes, inv_idx = process()
    with open('recipes.pkl', 'wb') as f:
        pickle.dump(recipes, f)
    with open('inv_idx.pkl', 'wb') as f:
        pickle.dump(inv_idx, f)
