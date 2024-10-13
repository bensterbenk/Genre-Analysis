import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (assuming the TSV file)
file_path = 'data/localify_music_genre.tsv'
df = pd.read_csv(file_path, sep='\t')

# Combine song name and artist into a unique identifier
df['song_id'] = df['song_name'] + ' - ' + df['artist_name']

# Group by song ID and count the number of distinct genres per song
multi_genre_songs = df.groupby('song_id')['genre_name'].nunique()

# Filter to keep only songs that have more than one genre
multi_genre_songs = multi_genre_songs[multi_genre_songs > 1]

# Get the subset of the dataframe that has these multi-genre songs
multi_genre_df = df[df['song_id'].isin(multi_genre_songs.index)]

# Create a new column combining genres per song
multi_genre_df['genre_combinations'] = multi_genre_df.groupby('song_id')['genre_name'].transform(lambda x: ', '.join(sorted(x.unique())))

# Count the most frequent genre combinations
genre_combinations_count = multi_genre_df.groupby('genre_combinations').size().reset_index(name='count')

# Sort by count to get the most common combinations
genre_combinations_count = genre_combinations_count.sort_values(by='count', ascending=False)

# Plot the most common genre combinations
plt.figure(figsize=(10, 6))
sns.barplot(x='count', y='genre_combinations', data=genre_combinations_count.head(10), palette='viridis')
plt.title('Top 10 Most Common Genre Combinations')
plt.xlabel('Number of Songs')
plt.ylabel('Genre Combinations')
plt.tight_layout()

# Show plot
plt.show()

# Display the total number of songs with multiple genres
total_multi_genre_songs = multi_genre_songs.shape[0]
print(f'Total number of songs with multiple genres: {total_multi_genre_songs}')

genre_counts = df.groupby('song_id')['genre_name'].nunique()

# Filter for songs with only one genre
single_genre_songs = genre_counts[genre_counts == 1].index

# Create a new dataframe that contains only songs with a single genre
single_genre_df = df[df['song_id'].isin(single_genre_songs)]

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
