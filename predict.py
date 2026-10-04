import pickle

with open("best_housing_price_model (3).pkl", "rb") as file:
    data = pickle.load(file)

print("Model loaded successfully!")
print("Model details:", data)