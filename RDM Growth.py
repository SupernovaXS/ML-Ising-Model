import numpy as np
import matplotlib.pyplot as plt
import random
import RFR3D as model

# Initialize parameters
size = 50  # Lattice size
iterations = 1500000  # Number of iterations
temperature = 1  # Temperature for Metropolis algorithm
thickness = 1  # Thickness of the material in cm
energies = [5, 50, 500, 5000, 50000]  # Different MeVs to test

# Initialize lattices
control_lattice = np.random.choice([-1, 1], size=(size, size))  # Control lattice

# Radiation site for experimental lattice
center_x, center_y = size // 2, size // 2

# Define the Metropolis function
def metropolis(lattice, temperature, absorbed_energy, radiation_site=None):
    if radiation_site:
        x, y = radiation_site
    else:
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
    
    spin = lattice[x, y]
    if radiation_site:  # Radiation perturbation
        delta_E = -spin * absorbed_energy
    else:  # Regular Ising interaction
        delta_E = -spin * np.sum([
            lattice[(x + dx) % size, (y + dy) % size] 
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ])
    
    # Metropolis criterion
    if delta_E < 0 or random.random() < np.exp(-delta_E / temperature):
        lattice[x, y] = -spin  # Flip the spin

    return lattice

def compute_flipped_fraction(control, experimental, center, radius):
    """Compute the fraction of flipped spins within a given radius of the radiation site."""
    x, y = center
    flipped_count = 0
    total_count = 0

    for i in range(control.shape[0]):
        for j in range(control.shape[1]):
            if (i - x)**2 + (j - y)**2 <= radius**2:  # Check within radius
                total_count += 1
                if control[i, j] != experimental[i, j]:
                    flipped_count += 1

    return flipped_count / total_count if total_count > 0 else 0

# Track damage for each energy level
damage_results = []

for energy in energies:
    # Initialize experimental lattice
    experimental_lattice = np.copy(control_lattice)  # Copy for experimental lattice

    # Use the ML model to predict coefficients
    coeff, trans, att = model.predict(energy, thickness)
    absorbed_energy = energy * att  # Absorbed energy at the radiation site

    # Print the input parameters
    print(f"Testing with Energy: {energy} MeV")
    print(f"Attenuation Coefficient: {coeff}")
    print(f"Transmission Percentage: {trans * 100}%")
    print(f"Attenuation Percentage: {att * 100}%")
    print(f"Absorbed Energy: {absorbed_energy:.2f} MeV\n")

    # Run the simulation for this energy
    for i in range(iterations):
        # Update control lattice (no radiation influence)
        control_lattice = metropolis(control_lattice, temperature, 0)
        
        # Update experimental lattice (radiation influence in the first half of iterations)
        if i < iterations // 2:
            experimental_lattice = metropolis(experimental_lattice, temperature, absorbed_energy, (center_x, center_y))
        else:
            experimental_lattice = metropolis(experimental_lattice, temperature, 0)

    # Calculate damage (flipped fraction) for this energy
    radius = 10  # Define a radius for local analysis
    flipped_fraction = compute_flipped_fraction(control_lattice, experimental_lattice, (center_x, center_y), radius)
    
    # Store the result
    damage_results.append(flipped_fraction)

# Print the results for radiation damage
print("\nRadiation Damage Growth Results:")
for energy, damage in zip(energies, damage_results):
    print(f"Energy: {energy} MeV, Flipped Fraction: {damage:.4f}")

# Plot the results for radiation damage growth
plt.figure(figsize=(8, 6))
plt.plot(energies, damage_results, marker='o', linestyle='-', color='b', label='Radiation Damage')
plt.title('Radiation Damage Growth with Increasing Energy (MeV)')
plt.xlabel('Energy (MeV)')
plt.ylabel('Fraction of Flipped Spins')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig("radiation_damage_growth.png")

# Show the plot
plt.show()
