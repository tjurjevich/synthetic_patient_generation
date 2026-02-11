# Define the architecture for the variational autoencoder and train model

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Layer

# Define numeric & categorical variables
num_vars = ["patient_age","patient_height_cm","patient_weight_kg","patient_systolic_bp","patient_diastolic_bp","patient_heart_rate"]
cat_vars = ["patient_gender","patient_race"]

# Encoder framework
class PatientEncoder(Layer):
    def __init__(self, edge_dim: int, latent_dim: int):
        super().__init__()
        # Layers for deterministic portion
        self.dense_1 = Dense(units = edge_dim, activation = "relu")
        self.dense_2 = Dense(units = edge_dim // 2, activation = "relu")
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
class PatientDecoder(Layer):
    def __init__(self, edge_dim: int, latent_dim: int, original_continuous_dimensions, num_categories_list):
        """
        original_continuous_dimensions: normalized, but unchanged continous variables
        num_categories_list: cardinality for each discrete variable (order of variables/cardinality will matter)
        """
        super().__init__()
        # Innermost dense layer
        self.dense_1 = Dense(units = latent_dim, activation = "relu")
        # Outermost dense layer
        self.dense_2 = Dense(units = edge_dim, activation = "relu")

        # Separate continuous from discrete variables
        self.output_continous = Dense(original_continuous_dimensions)
        self.output_discrete = [Dense(n, activation = "softmax") for n in num_categories_list]

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x_cont = self.output_continous(x)
        x_discrete = [layer(x) for layer in self.output_discrete]
        return tf.concat([x_cont] + x_discrete, axis = -1)

# Combined (VAE) framework
class PatientGenerator(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
    
    def call(self, inputs):
        mean, log_var, z = self.encoder(inputs)
        reconstructed_ouput = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_mean(
            log_var - tf.square(mean) - tf.exp(log_var) + 1
        )
        self.add_loss(kl_loss)
