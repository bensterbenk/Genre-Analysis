from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
from data_processing import remove_multigenre

file_path = "data/localify_music_genre.tsv" 

X_train, X_test, y_train, y_test  = remove_multigenre(file_path)

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

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                    factor=0.5, 
                                                    patience=5)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

# Use class_weights in model.fit()
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)

# Step 5: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_accuracy:.4f}")



train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_acc) + 1)

# Plot accuracy
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
plt.axvline(x=20, color='red', linestyle='--', label='Possible Divergence Start')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Zoom in on the area where divergence starts (around epoch 20)
plt.xlim(15, 30)  # Adjust as needed to zoom in
plt.ylim(min(val_acc[15:30]) - 0.05, max(train_acc[15:30]) + 0.05)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.axvline(x=20, color='red', linestyle='--', label='Possible Divergence Start')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Zoom in on the area where divergence starts (around epoch 20)
plt.xlim(15, 30)  # Adjust as needed to zoom in
plt.ylim(min(val_loss[15:30]) - 0.05, max(train_loss[15:30]) + 0.05)

plt.tight_layout()
plt.show()

genre_mapping_path = "data/localify_music_genre_key.tsv"  # Update with the actual file path
genre_mapping = pd.read_csv(genre_mapping_path, sep='\t')

# Create the genre_names list in the correct order
genre_names = genre_mapping['genre'].tolist()

# Generate predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Step 3: Normalize and Plot the Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot the normalized confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=genre_names, yticklabels=genre_names)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Genre')
plt.ylabel('True Genre')
plt.show()


# Step 2: Calculate and Plot Per-Class Precision and Recall
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)

# Create a bar plot for precision and recall
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(genre_names))

bar1 = ax.bar(index, precision, bar_width, label='Precision')
bar2 = ax.bar(index + bar_width, recall, bar_width, label='Recall')

ax.set_xlabel('Genres')
ax.set_ylabel('Scores')
ax.set_title('Per-Class Precision and Recall')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(genre_names, rotation=45)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()


train_genre_counts = train_data['genre_name'].value_counts()
print("Training Data Genre Counts:")
print(train_genre_counts)
print("\n")

# Genre counts in testing data
test_genre_counts = test_data['genre_name'].value_counts()
print("Testing Data Genre Counts:")
print(test_genre_counts)

report = classification_report(y_test, y_pred, target_names=genre_names)

# Print the classification report
print("Classification Report:")
print(report)

# Plotting feature distributions using box plots
plt.figure(figsize=(20, 30))

for feature in feature_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=feature, y='genre_name', data=data, orient='h')
    plt.title(f'Distribution of {feature} by Genre')
    plt.xlabel(feature)
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.show()


correlation_matrix = data[feature_columns + ['genre_encoded']].corr()

# Extract correlations with the target variable
target_correlations = correlation_matrix['genre_encoded'].drop('genre_encoded')

# Print the correlations with the target variable
print("Correlations between features and the target variable:")
print(target_correlations)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Alternatively, visualize only the correlations between features and the target
plt.figure(figsize=(8, 6))
sns.heatmap(target_correlations.to_frame(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation with Target Variable')
plt.xlabel('Genre Encoded')
plt.ylabel('Features')
plt.show()