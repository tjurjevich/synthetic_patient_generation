# Define the architecture for the variational autoencoder and train model
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Layer
import polars as pl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define numeric & categorical variables
NUM_VARS = ["patient_age","patient_height_cm","patient_weight_kg","patient_systolic_bp","patient_diastolic_bp","patient_heart_rate"]
CAT_VARS = ["patient_gender","patient_race"]

# Dimensions for inner/outer layers
EDGE_DIM = 128
LATENT_DIM = 16

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
        original_continuous_dimensions: normalized, but unchanged continous variables: 6 in this particular use-case
        num_categories_list: cardinality for each discrete variable (order of variables/cardinality will matter): [4, 7] in this particular use-case
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
class PatientDataGenerator(tf.keras.Model):
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
                tf.reduce_sum(tf.square(data[:, :self.num_continuous_variables] - reconstructed_output[:, :self.num_continuous_variables]), axis=1)
            )
            recon_cat = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    data[:, self.num_continuous_variables:], 
                    reconstructed_output[:, self.num_continuous_variables:], 
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
        
# Function which will engineer features from "original" data
class DataPreprocessor():
    def __init__(self, data: pl.DataFrame):
        self.data_ = data

    def preprocess(self, numeric_feature_names, categorical_feature_names):
        self.numeric_feature_names_ = numeric_feature_names
        self.categorical_feature_names_ = categorical_feature_names
        self.numeric_transformer_ = Pipeline([
            ("standard_scaler", StandardScaler())
        ])
        self.categorical_transformer_ = Pipeline([
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        self.preprocessor = ColumnTransformer([
            ("numeric", self.numeric_transformer_, numeric_feature_names),
            ("categorical", self.categorical_transformer_, categorical_feature_names)
        ])
        return self.preprocessor.fit_transform(self.data_.select(numeric_feature_names + categorical_feature_names))

if __name__ == "__main__":
    data = pl.read_parquet("./data/original_data.parquet")
    dp = DataPreprocessor(data = data)
    input_data = dp.preprocess(numeric_feature_names=NUM_VARS, categorical_feature_names=CAT_VARS)

    # Document field names and possible values, which will need to be used as upcoming parameters for model
    categorical_metadata = {}
    for i,cat in enumerate(CAT_VARS):
        possible_vals = dp.preprocessor.named_transformers_["categorical"].named_steps["one_hot_encoder"].categories_[i]
        categorical_metadata[cat] = possible_vals.tolist()

    # Initialize encoder & decoder
    encoder = PatientEncoder(edge_dim=EDGE_DIM, latent_dim=LATENT_DIM)
    decoder = PatientDecoder(edge_dim=EDGE_DIM, latent_dim=LATENT_DIM, original_continuous_dimensions=len(NUM_VARS), num_categories_list=[len(i) for i in categorical_metadata.values()])
    
    vae = PatientDataGenerator(encoder = encoder, decoder = decoder, num_continous_variables = len(NUM_VARS))


