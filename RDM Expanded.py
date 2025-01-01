import numpy as np
import matplotlib.pyplot as plt
import random
import RFR3D as model

# Initialize parameters
size = 50  # Lattice size
iterations = 1000000  # Number of iterations
temperature = 1  # Temperature for Metropolis algorithm
given_energy = 30000  # Energy of radiation in MeV
thickness = 1  # Thickness of the material in cm

# Initialize lattices
control_lattice = np.random.choice([-1, 1], size=(size, size))  # Control lattice
experimental_lattice = np.copy(control_lattice)  # Copy for experimental lattice

# Use the ML model to predict coefficients
coeff, trans, att = model.predict(given_energy, thickness)
absorbed_energy = given_energy * att  # Absorbed energy at the radiation site

print(f"Attenuation Coefficient: {coeff}")
print(f"Transmission Percentage: {trans * 100}%")
print(f"Attenuation Percentage: {att * 100}%\n")
print(f"Absorbed Energy: {absorbed_energy:.2f} MeV\n")

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
        delta_E = -spin * np.sum([lattice[(x + dx) % size, (y + dy) % size] for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]])
    
    # Metropolis criterion
    if delta_E < 0 or random.random() < np.exp(-delta_E / temperature):
        lattice[x, y] = -spin  # Flip the spin

    return lattice

# Radiation site for experimental lattice
center_x, center_y = size // 2, size // 2

# Run the simulation
total_energy_absorbed = 0  # Track total energy absorbed by the lattice
total_flipped_spins = 0  # Track the total number of flipped spins across the entire lattice

for i in range(iterations):
    # Update control lattice (no radiation influence)
    control_lattice = metropolis(control_lattice, temperature, 0)
    
    # Update experimental lattice (radiation influence in the first half of iterations)
    if i < iterations // 2:
        experimental_lattice = metropolis(experimental_lattice, temperature, absorbed_energy, (center_x, center_y))
        total_energy_absorbed += absorbed_energy  # Add to total energy absorbed
    else:
        experimental_lattice = metropolis(experimental_lattice, temperature, 0)
    
    # Count flipped spins across the entire lattice
    total_flipped_spins += np.sum(experimental_lattice != control_lattice)

# Plot the final lattices
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Control lattice plot
ax[0].imshow(control_lattice, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
ax[0].set_title("Control Lattice (No Radiation)")
ax[0].axis('off')

# Experimental lattice plot
ax[1].imshow(experimental_lattice, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
ax[1].set_title(f"Experimental Lattice (Radiation at {center_x}, {center_y})")
ax[1].axis('off')

# Add colorbars
for axis in ax:
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=axis, orientation='vertical', pad=0.02)
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(['-1 (Down)', '+1 (Up)'])

# Save and display the plots
plt.tight_layout()
plt.savefig("lattice_comparison.png")
plt.show()

# Quantitative Analysis Functions
def compute_magnetization(lattice):
    """Compute the average magnetization of a lattice."""
    return np.mean(lattice)

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

def compute_flipped_fraction_total(lattice, control_lattice):
    """Compute the fraction of flipped spins in the entire lattice."""
    total_count = lattice.size
    flipped_count = np.sum(lattice != control_lattice)  # Count flipped spins in the lattice
    return flipped_count / total_count

def spin_correlation(lattice, distance):
    """Compute the spin correlation function for a given distance."""
    size = lattice.shape[0]
    correlations = []

    for x in range(size):
        for y in range(size):
            x_offset = (x + distance) % size
            correlations.append(lattice[x, y] * lattice[x_offset, y])

    return np.mean(correlations)

# Metrics computation
control_mag = compute_magnetization(control_lattice)
experimental_mag = compute_magnetization(experimental_lattice)

radius = 10  # Define a radius for local analysis
center = (size // 2, size // 2)  # Radiation site at the center
flipped_fraction = compute_flipped_fraction(control_lattice, experimental_lattice, center, radius)

# New: Total flipped fraction comparison across entire lattice
flipped_fraction_control_total = compute_flipped_fraction_total(control_lattice, control_lattice)
flipped_fraction_experimental_total = compute_flipped_fraction_total(experimental_lattice, control_lattice)

correlation_control = spin_correlation(control_lattice, distance=1)
correlation_experimental = spin_correlation(experimental_lattice, distance=1)

# Print results
print(f"Control Magnetization: {control_mag}")
print(f"Experimental Magnetization: {experimental_mag}")
print(f"Fraction of Flipped Spins within radius {radius}: {flipped_fraction}")
print(f"Spin Correlation (Control, distance=1): {correlation_control}")
print(f"Spin Correlation (Experimental, distance=1): {correlation_experimental}")

# New metrics
print(f"Total Energy Absorbed by the System: {total_energy_absorbed:.2f} MeV")
print(f"Total Number of Flipped Spins: {total_flipped_spins}")
print(f"Control Lattice: Total Fraction of Flipped Spins: {flipped_fraction_control_total}")
print(f"Experimental Lattice: Total Fraction of Flipped Spins: {flipped_fraction_experimental_total}")
