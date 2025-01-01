import numpy as np
import matplotlib.pyplot as plt
import RFR3D as model

given_energy = 300
thickness = 1

dataset = model.predict(given_energy, thickness)
coeff = dataset[0]
trans = dataset[1]
att = dataset[2]

# print(coeff)
# print(trans)
# print(att)

print(f"The predicted attenuation coefficient for energy {given_energy} MeV is: {coeff}")
print(f"The predicted transmission for energy {given_energy} MeV and a thickness of {thickness} is {trans}.")
print(f"The predicted attenuation for energy {given_energy} MeV and a thickness of {thickness} is {att}.")
