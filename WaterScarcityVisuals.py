#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pygad
import numpy as np
import matplotlib.pyplot as plt

# Define the fitness function
def fitness_func(solution, solution_idx):
    # Example: Minimize the sum of squares (replace with actual optimization logic)
    return np.sum(solution ** 2)

# Define the parameters for the genetic algorithm
num_generations = 50
num_parents_mating = 5
sol_per_pop = 10
num_genes = 5  # Example: 5 different irrigation schedules

# Run the genetic algorithm
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=-4,
                       init_range_high=4)

ga_instance.run()

# Get the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution:", solution)
print("Best solution fitness:", solution_fitness)

# Plot the fitness evolution
plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Fitness Evolution')
plt.grid(True)

# Save the plot as an image
plt.savefig('optimization_results.png')
plt.show()


# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt

# Load example data (using MNIST dataset as a placeholder for satellite images)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Plot the training accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Save the plot as an image
plt.savefig('image_analysis_training.png')
plt.show()


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate synthetic data for demonstration purposes
years = np.arange(2000, 2051)
temperature = np.linspace(20, 25, len(years))  # Simulated temperature data
population = np.linspace(1, 2, len(years))  # Simulated population data (in millions)

# Simulated water usage based on temperature and population
water_usage = 100 + 10 * temperature + 50 * population + np.random.randn(len(years)) * 5

# Combine data into a DataFrame
data = pd.DataFrame({'Year': years, 'Temperature': temperature, 'Population': population, 'Water Usage': water_usage})

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train a linear regression model
X_train = train_data[['Temperature', 'Population']]
y_train = train_data['Water Usage']
model = LinearRegression()
model.fit(X_train, y_train)

# Predict water usage
X_test = test_data[['Temperature', 'Population']]
y_test = test_data['Water Usage']
predicted_usage = model.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(test_data['Year'], y_test, 'b', label='Actual')
plt.plot(test_data['Year'], predicted_usage, 'r--', label='Predicted')
plt.xlabel('Year')
plt.ylabel('Water Usage (million cubic meters)')
plt.title('Actual vs Predicted Water Usage')
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig('predictive_analytics.png')
plt.show()


# In[ ]:




