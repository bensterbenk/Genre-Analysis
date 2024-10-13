import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def remove_multigenre(file_path):

    # Step 1: Read the data
    data = pd.read_csv(file_path, sep='\t')

    # Step 2: Extract unique artists from the data
    artists = data['artist_name'].unique()

    # Step 3: Split the artists into training and testing sets
    train_artists, test_artists = train_test_split(
        artists,
        test_size=0.2,
        random_state=42
    )

    # Step 4: Create training data: all songs by artists in train_artists
    train_data = data[data['artist_name'].isin(train_artists)].reset_index(drop=True)

    # Step 5: Create testing data: all songs by artists in test_artists
    test_data = data[data['artist_name'].isin(test_artists)].reset_index(drop=True)

    # Ensure there's no overlap between train and test artist sets
    assert set(train_data['artist_name']).isdisjoint(set(test_data['artist_name'])), "Overlap detected in artists between train and test sets!"

    # Step 6: Process the training data to handle songs with multiple genres

    # Identify songs associated with multiple genres in the training data
    song_genre_counts = train_data.groupby(['song_name', 'artist_name'])['genre_name'].nunique().reset_index()
    song_genre_counts.rename(columns={'genre_name': 'genre_count'}, inplace=True)

    # Keep only songs that have a single genre
    single_genre_songs = song_genre_counts[song_genre_counts['genre_count'] == 1]

    # Merge back to the training data to filter
    train_data_single_genre = pd.merge(train_data, single_genre_songs[['song_name', 'artist_name']], on=['song_name', 'artist_name'], how='inner')

    # Optional: Drop duplicates if any exist after the merge
    train_data_single_genre = train_data_single_genre.drop_duplicates()

    # Replace train_data with the processed data
    train_data = train_data_single_genre

    # Proceed with the rest of your code using the modified train_data and the original test_data

    # Define feature columns
    feature_columns = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                    'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

    # Create a mapping from genre names to numeric codes
    genre_mapping = {
        'alternative': 0,
        'folk': 1,
        'rap': 2,
        'country': 3,
        'indie': 4,
        'soul': 5,
        'r&b': 6,
        'electronic': 7,
        'punk': 8,
        'jazz': 9
    }

    # Map genres to numeric codes in both train and test data
    train_data['genre_code'] = train_data['genre_name'].map(genre_mapping)
    test_data['genre_code'] = test_data['genre_name'].map(genre_mapping)

    # Handle any unmapped genres (if any)
    train_data = train_data.dropna(subset=['genre_code'])
    test_data = test_data.dropna(subset=['genre_code'])

    train_data = gaussian_noise(train_data)
    # Extract features and labels for training data
    X_train = train_data[feature_columns].values
    y_train = train_data['genre_code'].values.astype(int)

    # Extract features and labels for testing data
    X_test = test_data[feature_columns].values
    y_test = test_data['genre_code'].values.astype(int)

    print("y_train sample:", y_train[:5])
    print("y_test sample:", y_test[:5])
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Output the shapes of the datasets before and after SMOTE
    print("X_train shape before SMOTE:", X_train.shape)
    print("X_train shape after SMOTE:", X_train_resampled.shape)
    print("y_train shape after SMOTE:", y_train_resampled.shape)

    # Proceed with scaling
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    # Output the final shapes of the datasets to confirm the split
    print("X_train_resampled shape:", X_train_resampled.shape)
    print("X_test shape:", X_test.shape)
    print("y_train_resampled shape:", y_train_resampled.shape)
    print("y_test shape:", y_test.shape)

    return X_train_resampled, X_test, y_train_resampled, y_test


def gaussian_noise(df):
    numerical_features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                      'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

    # Define the noise level (e.g., 1% of the standard deviation)
    noise_level = 0.01

    # Create augmented data
    augmented_data = df.copy()
    for feature in numerical_features:
        std = df[feature].std()
        noise = np.random.normal(0, noise_level * std, size=df.shape[0])
        augmented_data[feature] += noise

    # Combine original and augmented data
    augmented_dataset = pd.concat([df, augmented_data], ignore_index=True)
    return augmented_dataset
