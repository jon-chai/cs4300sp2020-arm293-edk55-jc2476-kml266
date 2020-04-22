import pickle
from model import IngredientIndex

with open('model.pkl', 'rb') as f:
    Master = pickle.load(f)

print(Master.search(['apple', 'sage', 'chicken'], []))
