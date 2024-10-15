import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def process_main(file_path):

    df = pd.read_csv(file_path, sep='\t')

    # Combine song name and artist into a unique identifier
    df['song_id'] = df['song_name'] + ' - ' + df['artist_name']

    # Group by song ID and count the number of distinct genres per song
    genre_counts = df.groupby('song_id')['genre_name'].nunique()

    # Filter for songs with only one genre
    single_genre_songs = genre_counts[genre_counts == 1].index

    # Create a new dataframe that contains only songs with a single genre
    single_genre_df = df[df['song_id'].isin(single_genre_songs)]

    # Optionally, reset the index of the DataFrame
    single_genre_df = single_genre_df.reset_index(drop=True)

    # Calculate the original and new dataset sizes
    original_size = df.shape[0]
    new_size = single_genre_df.shape[0]
    reduction_size = original_size - new_size
    reduction_percentage = (reduction_size / original_size) * 100

    # Display results
    print(f'Original dataset size: {original_size} rows')
    print(f'New dataset size with only single-genre songs: {new_size} rows')
    print(f'Dataset reduction: {reduction_size} rows')
    print(f'Dataset reduction percentage: {reduction_percentage:.2f}%')

    # Define the column names (adjust if necessary)
    column_names = [
        'song_name', 'artist_name', 'acousticness', 'danceability', 'energy',
        'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo',
        'valence', 'genre_name', 'target'  # Assuming 'target' is your target variable
    ]

    # Read the TSV file into a DataFrame
    df = pd.read_csv(file_path, sep='\t', names=column_names, header=0)

    # Select feature columns (excluding non-numeric ones like 'song_name' and 'artist_name')
    feature_columns = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
    ]

    # Prepare the feature matrix X
    X = df[feature_columns].values

    # Prepare the target vector y
    # If 'genre_name' is your target variable and it's categorical, encode it
    le = LabelEncoder()
    y = le.fit_transform(df['genre_name'])

    # Alternatively, if 'target' is your target variable and it's numerical
    # y = df['target'].values

    # Calculate the index for the 80/20 split
    split_index = int(len(X) * 0.8)

    # Split the data into training and testing sets
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test

    


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


def separate_single_and_multi_genre(data):
    # Group by song_name and artist_name to count unique genres per song
    genre_counts = data.groupby(['song_name', 'artist_name'])['genre_name'].nunique().reset_index()
    genre_counts.rename(columns={'genre_name': 'genre_count'}, inplace=True)
    
    # Merge the counts back to the original data
    data_with_counts = pd.merge(data, genre_counts, on=['song_name', 'artist_name'], how='left')
    
    # Separate single-genre and multi-genre songs
    single_genre_data = data_with_counts[data_with_counts['genre_count'] == 1].copy()
    multi_genre_data = data_with_counts[data_with_counts['genre_count'] > 1].copy()
    
    # Drop the 'genre_count' column as it's no longer needed
    single_genre_data.drop(columns=['genre_count'], inplace=True)
    multi_genre_data.drop(columns=['genre_count'], inplace=True)
    
    return single_genre_data, multi_genre_data


def split_single_genre_data(single_genre_data):
    # Use train_test_split to split single-genre data
    train_data, test_data = train_test_split(
        single_genre_data,
        test_size=0.2,
        random_state=42,
        stratify=single_genre_data['genre_name']
    )
    return train_data, test_data



def process_data(train_data, test_single_genre_data, test_multi_genre_data):
    # Combine the test sets
    test_data = pd.concat([test_single_genre_data, test_multi_genre_data], ignore_index=True)
    
    # Define feature columns
    feature_columns = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                       'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    
    # Map genres to numeric codes
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
    train_data['genre_code'] = train_data['genre_name'].map(genre_mapping)
    test_data['genre_code'] = test_data['genre_name'].map(genre_mapping)
    
    # Handle unmapped genres (if any)
    train_data.dropna(subset=['genre_code'], inplace=True)
    test_data.dropna(subset=['genre_code'], inplace=True)
    
    # Extract features and labels
    X_train = train_data[feature_columns].values
    y_train = train_data['genre_code'].values.astype(int)
    
    X_test = test_data[feature_columns].values
    y_test = test_data['genre_code'].values.astype(int)
    
    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_resampled, y_test
