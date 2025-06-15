import numpy as np
import matplotlib.pyplot as plt

def create_gaussian_distribution(size):
    # Generate a normal distribution
    rng = np.random.default_rng()
    dist = rng.normal(0, 1, size)
    dist.sort()  # To create a gaussian curve like array
    dist = np.abs(dist)  # Flip negative values "up"
    # Normalize the distribution to sum to one
    dist /= dist.sum()
    # Ensure the sum is exactly one
    # sm = dist.sum()
    # if sm != 1.0:
    #     idx = rng.choice(size)  # Choose a random index
    #     dist[idx] += 1 - sm
    return dist

# Example usage
nr_options = 20
gaussian_dist = create_gaussian_distribution(nr_options)
s = gaussian_dist.sum()

nr_zeroes = gaussian_dist.size - np.count_nonzero(gaussian_dist)
print("There are", nr_zeroes, "zero values in the distribution")

# Plot the distribution
plt.plot(gaussian_dist)
plt.title("Normalized Gaussian Distribution")
plt.show()

sample_size = 800
pool = np.arange(nr_options)
rng = np.random.default_rng()
print(pool.shape)
chosen = rng.choice(pool, sample_size, p=gaussian_dist)

plt.hist(chosen)
plt.show()