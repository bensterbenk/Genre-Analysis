from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

#EDITING DATASET SO THAT WE HANDLE GENRE AS A MULTIPLE LABEL BINARIZER

df = pd.read_csv('data/localify_music_genre.tsv', sep='\t')

#store the original index so we can keep order
df['original_index'] = df.index

# Groups multiple entries with the same song name and artist name into one entry with multiple genres
df_grouped = df.groupby(['song_name', 'artist_name']).agg({
    'acousticness': 'first',
    'danceability': 'first',
    'energy': 'first',
    'instrumentalness': 'first',
    'liveness': 'first',
    'loudness': 'first',
    'speechiness': 'first',
    'tempo': 'first',
    'valence': 'first',
    'genre_name': lambda x: list(x.unique()),
    'original_index': 'first'
}).reset_index()

# Maintains order by sorting newly grouped dataset by the original index
df_grouped = df_grouped.sort_values('original_index').reset_index(drop=True)

# Get rid of the original index 
df_grouped = df_grouped.drop(columns=['original_index'])



#THE ARTIST FILTER
unique_artists = df_grouped['artist_name'].unique()

# Split the artists into training (80%) and test (20%) sets
artists_train, artists_test = train_test_split(unique_artists, test_size=0.22, random_state=42)

# Filter the original grouped DataFrame by artists
train_data = df_grouped[df_grouped['artist_name'].isin(artists_train)].reset_index(drop=True)
test_data = df_grouped[df_grouped['artist_name'].isin(artists_test)].reset_index(drop=True)

# Prepare features X and labels y for training and testing sets
X_train = train_data[['acousticness', 'danceability', 'energy', 'instrumentalness',
                      'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
X_test = test_data[['acousticness', 'danceability', 'energy', 'instrumentalness',
                    'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]

# Print the number of artists in each split
print(f"Number of artists in training set: {len(artists_train)}")
print(f"Number of artists in test set: {len(artists_test)}")

# Print the number of samples in each split
print(f"Number of songs in training data: {train_data.shape[0]}")
print(f"Number of songs in test data: {test_data.shape[0]}")

print(f"Training Data %: {train_data.shape[0]/(train_data.shape[0]+test_data.shape[0])}")


# Prepare labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_data['genre_name'])
y_test = mlb.transform(test_data['genre_name'])  # Use transform here to ensure consistency

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Output the shapes of the datasets to confirm the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#THE MODEL
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(mlb.classes_), activation='sigmoid')
])

#using binary accuracy since it's a better metric for mlb
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])


history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

test_loss, test_binary_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Binary Accuracy: {test_loss:.4f}")
print(f"Test Binary Accuracy: {test_binary_accuracy:.4f}")

# Plot accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model Binary Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()