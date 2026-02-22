# Define the architecture for the variational autoencoder and train model
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Layer
import polars as pl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define numeric & categorical variables
NUM_VARS = ["patient_age","patient_height_cm","patient_weight_kg","patient_systolic_bp","patient_diastolic_bp","patient_heart_rate"]
CAT_VARS = ["patient_gender","patient_race"]

# Dimensions for inner/outer layers
ENCODER_OUTER_DIM = 256
DECODER_INNER_DIM = 128
LATENT_DIM = 8
BETA = 6
FREE_BITS = 2.0
EPOCHS = 10

# Encoder framework
class PatientEncoder(Layer):
    def __init__(self, outer_dim: int, latent_dim: int):
        super().__init__()
        # Layers for deterministic portion
        self.dense_1 = Dense(units = outer_dim, activation = "relu")
        self.batch_norm_1 = BatchNormalization()
        self.dense_2 = Dense(units = outer_dim // 2, activation = "relu")
        self.batch_norm_2 = BatchNormalization()
        # Layers for reparameterization
        self.z_mean = Dense(units = latent_dim, name = "z_mean")
        self.z_log_var = Dense(units = latent_dim, name = "z_log_var")
    
    def reparameterization(self, mean, log_var):
        epsilon = tf.random.normal(shape = tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon
    
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.batch_norm_1(x)
        x = self.dense_2(x)
        x = self.batch_norm_2(x)
        mean = self.z_mean(x)
        log_var = self.z_log_var(x)
        z = self.reparameterization(mean = mean, log_var = log_var)
        return mean, log_var, z
        
# Decoder framework
class PatientDecoder(Layer):
    def __init__(self, inner_dim: int, num_numeric_dimensions, categorical_cardinalities):
        """
        original_continuous_dimensions: normalized, but unchanged continous variables: 6 in this particular use-case
        num_categories_list: cardinality for each discrete variable (order of variables/cardinality will matter): [4, 7] in this particular use-case
        """
        super().__init__()
        # Innermost dense layer
        self.dense_1 = Dense(units = inner_dim, activation = "relu")
        # Outermost dense layer
        self.dense_2 = Dense(units = inner_dim * 2, activation = "relu")

        # Separate continuous from discrete variables
        self.output_numeric = Dense(num_numeric_dimensions)
        self.output_categorical = [Dense(n, activation = "softmax") for n in categorical_cardinalities]

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x_num = self.output_numeric(x)
        x_cat = [layer(x) for layer in self.output_categorical]
        return tf.concat([x_num] + x_cat, axis = -1)

# Combined (VAE) framework
class PatientDataGenerator(tf.keras.Model):
    def __init__(self, encoder, decoder, num_numeric_dimensions):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.num_continuous_variables = num_numeric_dimensions

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
            recon_num = tf.reduce_mean(
                tf.reduce_sum(tf.square(data[:, :self.num_continuous_variables] - reconstructed_output[:, :self.num_continuous_variables]), axis=1)
            )

            recon_cat = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    data[:, self.num_continuous_variables:], 
                    reconstructed_output[:, self.num_continuous_variables:], 
                    from_logits=True,
                    label_smoothing=0.1
                )
            )

            # 4. KL divergence loss
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            kl_loss = tf.maximum(kl_loss, FREE_BITS)  # caps KL loss to a minimum of FREE_BITS parameter
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # 5. Add reconstruction losses with KL divergence loss for overall loss
            total_loss = recon_cat + recon_num + (BETA * kl_loss)
            
            # 6. Calculate gradients and apply to current set of model weights
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            # 7. Update metrics
            self.reconstruction_loss_tracker.update_state(recon_cat + recon_num)
            self.kl_loss_tracker.update_state(kl_loss)
            self.total_loss_tracker.update_state(total_loss)
            return {m.name: m.result() for m in self.metrics}



# Function which will engineer features from "original" data
class DataPreprocessor():
    def __init__(self, data: pl.DataFrame):
        self.data_ = data

    def transform(self, numeric_feature_names, categorical_feature_names):
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
        # Run fit() method separate from transform() method, which will return the actual values.
        self.preprocessor.fit(self.data_.select(numeric_feature_names + categorical_feature_names))
        return self.preprocessor.transform(self.data_.select(numeric_feature_names + categorical_feature_names))

    def inverse_transform(self, raw_data):
        return self.preprocessor.inverse_transform(raw_data)

def process_categorical_model_output(preprocessor: DataPreprocessor, raw_output: np.array, categorical_feature_names):
    cleaned_arrays = []
    for field in categorical_feature_names:
        slice_indices = [i for i,j in enumerate(preprocessor.preprocessor.get_feature_names_out()) if field in j]
        sliced_data = raw_output[:, min(slice_indices):max(slice_indices)+1]
        cat_field_array = np.zeros_like(sliced_data)
        cat_field_array[np.arange(len(sliced_data)), np.argmax(sliced_data, axis = 1)] = 1
        cleaned_arrays.append(cat_field_array)
    cleaned_arrays = tuple(cleaned_arrays)
    binarized_data = np.hstack(cleaned_arrays)
    return dp.preprocessor.named_transformers_["categorical"].inverse_transform(binarized_data)


if __name__ == "__main__":
    data = pl.read_parquet("./data/original_data.parquet")
    dp = DataPreprocessor(data = data)
    input_data = dp.transform(numeric_feature_names=NUM_VARS, categorical_feature_names=CAT_VARS)

    # class_weights_list = []
    # ohe = dp.preprocessor.named_transformers_["categorical"].named_steps["one_hot_encoder"]
    # for i, cat in enumerate(CAT_VARS):
    #     categories = ohe.categories_[i]
    #     y = data[cat].to_numpy()
    #     weights = compute_class_weight("balanced", classes=categories, y=y)
    #     class_weights_list.append(weights)

    # Concatenate into a single weight vector matching the one-hot encoded columns
    # cat_class_weights = np.concatenate(class_weights_list).astype(np.float32)
    # cat_class_weights_tf = tf.constant(cat_class_weights)

    # Document field names and possible values, which will need to be used as upcoming parameters for model
    categorical_metadata = {}
    for i,cat in enumerate(CAT_VARS):
        possible_vals = dp.preprocessor.named_transformers_["categorical"].named_steps["one_hot_encoder"].categories_[i]
        categorical_metadata[cat] = possible_vals.tolist()

    # Initialize encoder & decoder
    encoder = PatientEncoder(outer_dim=ENCODER_OUTER_DIM, latent_dim=LATENT_DIM)
    decoder = PatientDecoder(inner_dim=DECODER_INNER_DIM, num_numeric_dimensions=len(NUM_VARS), categorical_cardinalities=[len(i) for i in categorical_metadata.values()])
    
    # Pass encoder & decoder into PatientDataGenerator object
    vae = PatientDataGenerator(encoder = encoder, decoder = decoder, num_numeric_dimensions = len(NUM_VARS))

    # Compile and train
    vae.compile(optimizer = "adam")
    vae.fit(
        input_data,
        epochs = EPOCHS,
        batch_size = 256
    )

    # Sample new points from normal dist (prior)
    z_new = tf.random.normal(shape = (100, LATENT_DIM))

    # Decode to original input dimensions
    unprocessed_output = vae.decoder(z_new)

    processed_numeric_output = dp.preprocessor.named_transformers_["numeric"].inverse_transform(unprocessed_output.numpy()[:, :len(NUM_VARS)])
    processed_categorical_output = process_categorical_model_output(dp, raw_output=unprocessed_output, categorical_feature_names=CAT_VARS)
    
    # print(processed_numeric_output.shape)
    # print(processed_categorical_output.shape)
    print(np.hstack((processed_numeric_output, processed_categorical_output)))
