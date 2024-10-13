from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from utils import artist_split, mlb_izer, class_weights
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score

df_grouped = mlb_izer("data/localify_music_genre.tsv")

X_train, X_test, y_train, y_test, mlb = artist_split(df_grouped)

class_weights = class_weights(y_train)

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
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation='softmax')
])


#using binary accuracy since it's a better metric for mlb
optimizer = tf.keras.optimizers.Adam()  # Uses the default learning rate of 0.001
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')])


history = model.fit(
    X_train, y_train,
    epochs=100 ,
    batch_size=32,
    validation_data=(X_test, y_test)
)

test_loss, test_binary_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_binary_accuracy:.4f}")


# Predict probabilities on the test set
y_pred_prob = model.predict(X_test)

# Binarize the predictions with a threshold of 0.5
y_pred = (y_pred_prob >= 0.5).astype(int)

# Ensure at least one genre is predicted per sample
for i in range(y_pred.shape[0]):
    if y_pred[i].sum() == 0:
        # Set the genre with the highest probability to 1
        y_pred[i, np.argmax(y_pred_prob[i])] = 1


# Plot AUC
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Model AUC')
plt.ylabel('AUC')
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

for i, genre in enumerate(mlb.classes_):
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_prob[:, i])
    average_precision = average_precision_score(y_test[:, i], y_pred_prob[:, i])
    plt.figure()
    plt.step(recall, precision, where='post', label=f'AP = {average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve for Genre: {genre}')
    plt.legend()
    plt.show()
