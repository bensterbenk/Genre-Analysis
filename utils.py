from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


#EDITING DATASET SO THAT WE HANDLE GENRE AS A MULTIPLE LABEL BINARIZER
def mlb_izer(data_path):

    df = pd.read_csv(data_path, sep='\t')

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
    return df_grouped


def artist_split(df_grouped):

    #THE ARTIST FILTER
    unique_artists = df_grouped['artist_name'].unique()

    # Split the artists into training (80%) and test (20%) sets
    artists_train, artists_test = train_test_split(unique_artists, test_size=0.22, random_state=42)

    # Filter the original grouped DataFrame by artists
    train_data = df_grouped[df_grouped['artist_name'].isin(artists_train)].reset_index(drop=True)
    test_data = df_grouped[df_grouped['artist_name'].isin(artists_test)].reset_index(drop=True)

    print(f"Training Data %: {train_data.shape[0]/(train_data.shape[0]+test_data.shape[0])}")

    # Prepare features X and labels y for training and testing sets
    X_train = train_data[['acousticness', 'danceability', 'energy', 'instrumentalness',
                        'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
    X_test = test_data[['acousticness', 'danceability', 'energy', 'instrumentalness',
                        'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]

    
    # Prepare labels using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_data['genre_name'])
    y_test = mlb.transform(test_data['genre_name'])  # Use transform here to ensure consistency
    return X_train, X_test, y_train, y_test, mlb
