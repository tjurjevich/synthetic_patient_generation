# Define the architecture for the variational autoencoder and train model

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization

# Define numeric & categorical variables
num_vars = ["patient_age","patient_height_cm","patient_weight_kg","patient_systolic_bp","patient_diastolic_bp","patient_heart_rate"]
cat_vars = ["patient_gender","patient_race"]

# Encoder framework
class PatientEncoder(tf.keras.Model):
    def __init__(self, edge_dim: int, latent_dim: int):
        super().__init__()
        # Layers for deterministic portion
        self.dense_1 = Dense(units = edge_dim, activation = "relu")
        self.dense_2 = Dense(units = edge_dim, activation = "relu")
        # Layers for reparameterization
        self.z_mean = Dense(units = latent_dim, name = "z_mean")
        self.z_log_var = Dense(units = latent_dim, name = "z_log_var")
    
    def reparameterization(self, mean, log_var):
        epsilon = tf.random.normal(shape = tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon
    
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        mean = self.z_mean(x)
        log_var = self.z_log_var(x)
        z = self.reparameterization(mean = mean, log_var = log_var)
        return mean, log_var, z
        


# Decoder framework


# Combined (VAE) framework
