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
    def __init__(self, encoder, decoder, num_continous_variables):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.num_continuous_variables = num_continous_variables
    
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property 
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        mean, log_var, z = self.encoder(inputs)
        reconstructed_ouput = self.decoder(z)
        return reconstructed_ouput
    
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            # 1. Encoder = forward pass
            mean, log_var, z = self.encoder(data)
            # 2. Decoder = forward pass
            reconstructed_output = self.decoder(z)

            # 3. Split reconstruction into continuous and categorical parts
            recon_cont = tf.reduce_mean(
                tf.reduce_sum(tf.square(data[:, :self.num_cont] - reconstructed_output[:, :self.num_cont]), axis=1)
            )
            recon_cat = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    data[:, self.num_cont:], 
                    reconstructed_output[:, self.num_cont:], 
                    from_logits=True
                )
            )
            # 4. KL divergence loss
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis = 1))

            # 5. Add reconstruction losses with KL divergence loss for overall loss
            total_loss = recon_cat + recon_cont + kl_loss
            
            # 6. Calculate gradients and apply to current set of model weights
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            # 7. Update metrics
            self.reconstruction_loss_tracker.update_state(recon_cat + recon_cont)
            self.kl_loss_tracker.update_state(kl_loss)
            self.total_loss_tracker.update_state(total_loss)
            return {m.name: m.result() for m in self.metrics}