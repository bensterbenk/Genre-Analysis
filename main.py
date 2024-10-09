from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from utils import artist_split, mlb_izer

df_grouped = mlb_izer("data/localify_music_genre.tsv")

X_train, X_test, y_train, y_test, mlb = artist_split(df_grouped)

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
print(f"Test ßLoss: {test_loss:.4f}")
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

# Predict probabilities on the test set
y_pred_prob = model.predict(X_test)

# Binarize the predictions with a threshold of 0.5
y_pred = (y_pred_prob >= 0.5).astype(int)

# Generate confusion matrices for each genre
confusion_matrices = multilabel_confusion_matrix(y_test, y_pred)

# Plot the confusion matrix for each genre
for i, genre in enumerate(mlb.classes_):
    plt.figure(figsize=(5, 5))
    plt.imshow(confusion_matrices[i], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Genre: {genre}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Not Present', 'Present'], rotation=45)
    plt.yticks(tick_marks, ['Not Present', 'Present'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
