import pickle
import sys

import pandas as pd

conditions = {
    0: "HEALTHY",
    1: "INNER RING DAMAGE",
    2: "OUTER RING DAMAGE"
}

with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

data = pd.read_csv(sys.argv[1])
prediction = model.predict(data)[0]

print("Bearing condition:", conditions[prediction])
