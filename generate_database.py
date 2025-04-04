import numpy as np
import pandas as pd


def generate_database(numdatapoint=100, noise_level=5.0):
    # Lists to store data for each column
    x_initial_positions = []
    y_initial_positions = []
    x_initial_velocities = []
    y_initial_velocities = []
    times = []
    gravities = []
    x_final_positions = []
    y_final_positions = []

    # Generate data for each data point
    for datapoint in range(numdatapoint):
        # Random initial conditions for each data point
        x_initial_position = np.random.randint(1, 100)  # Random x initial position (1 to 99)
        y_initial_position = np.random.randint(1, 100)  # Random y initial position (1 to 99)
        x_initial_velocity = np.random.randint(10, 60)  # Random x initial velocity (10 to 59 m/s)
        y_initial_velocity = np.random.randint(10, 60)  # Random y initial velocity (10 to 59 m/s)
        t = np.random.randint(5, 60)                    # Random time (5 to 59 seconds)
        g = 9.8 / 2                                     # Gravity (half of Earth’s, ~4.9 m/s²)

        # Calculate final positions based on projectile motion equations
        x_final_position = x_initial_position + (t * x_initial_velocity)  # No acceleration in x
        y_final_position = y_initial_position + (t * y_initial_velocity) - (g * t * t)  # Includes gravity effect

        # Add random Gaussian noise to final positions
        x_noise = np.random.normal(0, noise_level)  # Mean = 0, standard deviation = noise_level
        y_noise = np.random.normal(0, noise_level)
        x_final_position += x_noise
        y_final_position += y_noise

        # Append data to lists
        x_initial_positions.append(x_initial_position)
        y_initial_positions.append(y_initial_position)
        x_initial_velocities.append(x_initial_velocity)
        y_initial_velocities.append(y_initial_velocity)
        times.append(t)
        gravities.append(g)
        x_final_positions.append(x_final_position)
        y_final_positions.append(y_final_position)

    # Create a DataFrame from the collected data
    data = {
        'x_initial_position': x_initial_positions,
        'y_initial_position': y_initial_positions,
        'x_initial_velocity': x_initial_velocities,
        'y_initial_velocity': y_initial_velocities,
        'time': times,
        'gravity': gravities,
        'x_final_position': x_final_positions,
        'y_final_position': y_final_positions
    }
    df = pd.DataFrame(data)

    return df


# Example usage
if __name__ == "__main__":
    # Generate database with 100 data points and noise level of 5.0
    database = generate_database(numdatapoint=1000, noise_level=5.0)

    # Display the first few rows
    # print(database.head())

    # Optionally save to CSV
    database.to_csv('data/projectile_data.csv', index=False)
    print("Database saved to 'projectile_data.csv'")
