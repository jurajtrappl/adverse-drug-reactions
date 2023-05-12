import codecs
import pandas as pd
from SmilesPE.tokenizer import SPE_Tokenizer
import tensorflow as tf

SEED = 42

tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# ---------
# - Data -
# ---------

# Read input data.
smiles_embedding_df = pd.read_csv('./smiles_embedding.csv')

# Prepare training and validation data.
X_smiles = smiles_embedding_df['ISO_SMILE'].sample(n=len(smiles_embedding_df), random_state=SEED)

# ----------------
# - Tokenization -
# ----------------

# Tokenize to sequences and pad them to equal length.
spe_vob = codecs.open('./models/SPE_ChEMBL.txt')
spe = SPE_Tokenizer(spe_vob)

X_train_smiles_tokenized = [spe.tokenize(smile).split() for smile in X_smiles]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train_smiles_tokenized)

smiles_sequences = tokenizer.texts_to_sequences(X_train_smiles_tokenized)

max_seq_len = max(len(seq) for seq in smiles_sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(smiles_sequences, maxlen=max_seq_len, padding='post')

# -----------
# - Encoder -
# -----------

# One input is just a padded sequence.
encoder_input = tf.keras.layers.Input(shape=(max_seq_len,))

l2 = tf.keras.regularizers.L2(l2=0.01)

vocabulary_size = len(tokenizer.word_index)
embedding_size = 50
encoder_embedding_layer = tf.keras.layers.Embedding(
    input_dim=vocabulary_size + 1,
    output_dim=embedding_size,
    input_length=max_seq_len,
    mask_zero=True
)(encoder_input)

# The purpose of this LSTM layer is to model the temporal dependencies between the tokens in the input sequence.
encoder_lstm = tf.keras.layers.LSTM(256, return_sequences=True, kernel_regularizer=l2)
encoder_rnn = tf.keras.layers.Bidirectional(encoder_lstm, merge_mode='ave')(encoder_embedding_layer)

# Take max of each dimension.
encoder_pooled = tf.keras.layers.GlobalMaxPooling1D()(encoder_rnn)

encoder_model = tf.keras.Model(inputs=encoder_input, outputs=encoder_pooled)

# -----------
# - Decoder -
# -----------
decoder_input = tf.keras.layers.Input(shape=(embedding_size,))
decoder_output = tf.keras.layers.Dense(units=max_seq_len, kernel_regularizer=l2)(decoder_input)

decoder_model = tf.keras.Model(inputs=decoder_input, outputs=decoder_output)

# ---------------
# - Autoencoder -
# ---------------
autoencoder_input = encoder_input
autoencoder_output = decoder_model(encoder_pooled)

autoencoder_model = tf.keras.Model(inputs=autoencoder_input, outputs=autoencoder_output)

autoencoder_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError(name="loss"),
)

autoencoder_model.summary()

# --------------------
# - Network training -
# --------------------
EPOCHS = 50
BATCH_SIZE = 1000

def create_dataset(data):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(lambda smile: (smile, smile))
    dataset = dataset.shuffle(buffer_size=len(data), seed=SEED)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return dataset
    

train = create_dataset(padded_sequences)

autoencoder_model.fit(
    train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[tf.keras.callbacks.d(filepath='we_network_{epoch}.h5'), tf.keras.callbacks.TensorBoard('./logs')]
)
