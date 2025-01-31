import numpy as np
import matplotlib.pyplot as plt
import RFR3D as model

def initialize_lattice(size):
    """Initialize a lattice with random spins (-1 or 1)."""
    return np.random.choice([-1, 1], size=(size, size))

def calculate_energy(lattice, J=1):
    """Calculate the total energy of the lattice."""
    energy = 0
    size = lattice.shape[0]
    for i in range(size):
        for j in range(size):
            spin = lattice[i, j]
            neighbors = (
                lattice[(i + 1) % size, j] +
                lattice[i, (j + 1) % size] +
                lattice[(i - 1) % size, j] +
                lattice[i, (j - 1) % size]
            )
            energy -= J * spin * neighbors
    return energy / 2  # Avoid double counting

def monte_carlo_step(lattice, temperature, J=1):
    """Perform one Monte Carlo step."""
    size = lattice.shape[0]
    for _ in range(size * size):
        x, y = np.random.randint(0, size, size=2)
        spin = lattice[x, y]
        neighbors = (
            lattice[(x + 1) % size, y] +
            lattice[x, (y + 1) % size] +
            lattice[(x - 1) % size, y] +
            lattice[x, (y - 1) % size]  # Correct indexing
        )
        delta_energy = 2 * J * spin * neighbors
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            lattice[x, y] *= -1

def magnetization(lattice):
    """Calculate normalized magnetization."""
    max_magnetization = lattice.size  # All spins aligned
    return np.abs(np.sum(lattice)) / max_magnetization

def apply_radiation(lattice, attenuation_coeff):
    """Apply radiation by flipping spins based on attenuation probability."""
    size = lattice.shape[0]
    for i in range(size):
        for j in range(size):
            if np.random.rand() < attenuation_coeff:
                lattice[i, j] *= -1  # Flip the spin

def simulate_lattice(size, temperatures, attenuation_coeff=None):
    """Simulate the Ising model for control or experimental lattice."""
    lattice = initialize_lattice(size)
    magnetizations = []

    for temp in temperatures:
        for _ in range(100):  # Multiple Monte Carlo steps for equilibrium
            monte_carlo_step(lattice, temp)
        if attenuation_coeff:
            apply_radiation(lattice, attenuation_coeff)
        magnetizations.append(magnetization(lattice))

    return magnetizations

# Parameters
lattice_size = 50
temperatures = np.linspace(0.1, 5.0, 100)
given_energy = 200  # Example MeV value
thickness_cm = 1  # Example thickness in cm

# Get attenuation data
attenuation_data = model.predict(given_energy, thickness_cm)
attenuation_coeff = attenuation_data[0]  # Extract the attenuation coefficient

# Simulations
control_magnetizations = simulate_lattice(lattice_size, temperatures)
experimental_magnetizations = simulate_lattice(lattice_size, temperatures, attenuation_coeff)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(temperatures, control_magnetizations, label="Control Lattice", color="blue")
plt.plot(temperatures, experimental_magnetizations, label="Experimental Lattice", color="red")
plt.title("Magnetization vs Temperature")
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetization (|M|)")
plt.legend()
plt.grid()
plt.savefig("mag.png")
plt.show()
