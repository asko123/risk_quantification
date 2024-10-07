import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Data Collection and Preparation
def collect_and_prepare_data(num_samples=1000):
    # Simulating data for demonstration purposes
    impact = np.random.exponential(scale=50, size=num_samples)
    likelihood = np.random.beta(2, 5, size=num_samples)
    data = np.column_stack((impact, likelihood))
    return data

# Generative Model (GAN)
class RiskGAN:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(2, activation='sigmoid')
        ])
        return model

    def build_discriminator(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(2,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        model = keras.Sequential([
            self.generator,
            self.discriminator
        ])
        return model

    def train(self, data, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, data.shape[0], batch_size)
            real = data[idx]
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            fake = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

    def generate_scenarios(self, num_scenarios):
        noise = np.random.normal(0, 1, (num_scenarios, self.input_dim))
        return self.generator.predict(noise)

# Monte Carlo Simulation
def monte_carlo_simulation(gan, num_simulations=10000):
    scenarios = gan.generate_scenarios(num_simulations)
    risk_scores = np.array([calculate_risk_score(impact, likelihood) 
                            for impact, likelihood in scenarios])
    return risk_scores

# Risk Scoring Framework
def calculate_risk_score(impact, likelihood):
    # Simple risk scoring function
    return impact * likelihood * 100  # Scale up for visibility

# Visualization
def visualize_risk_distribution(risk_scores):
    kde = gaussian_kde(risk_scores)
    x_range = np.linspace(min(risk_scores), max(risk_scores), 100)
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, kde(x_range))
    plt.title('Risk Score Distribution')
    plt.xlabel('Risk Score')
    plt.ylabel('Density')
    plt.show()

# Main execution
def main():
    # Data preparation
    data = collect_and_prepare_data()

    # GAN training
    gan = RiskGAN(input_dim=10)  # Using 10-dimensional noise input
    gan.train(data, epochs=5000)

    # Monte Carlo simulation
    risk_scores = monte_carlo_simulation(gan)

    # Visualization
    visualize_risk_distribution(risk_scores)

    # Basic statistics
    print(f"Mean Risk Score: {np.mean(risk_scores)}")
    print(f"Median Risk Score: {np.median(risk_scores)}")
    print(f"95th Percentile Risk Score: {np.percentile(risk_scores, 95)}")

if __name__ == "__main__":
    main()
