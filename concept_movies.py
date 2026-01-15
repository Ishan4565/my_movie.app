import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

print("="*80)
print("MOVIE RECOMMENDATION SYSTEM - DECODED & FIXED")
print("="*80)

# --- STEP 1: DATASET ---
movies_data = {
    'title': [
        'The Dark Knight', 'Inception', 'Interstellar', 'The Avengers', 'Iron Man',
        'The Matrix', 'John Wick', 'Toy Story', 'Finding Nemo', 'The Lion King',
        'Pulp Fiction', 'The Godfather', 'Forrest Gump', 'The Shawshank Redemption',
        'Fight Club', 'Goodfellas', 'The Departed', 'Casino Royale', 'Skyfall',
        'Mission Impossible'
    ],
    'genre': [
        'Action Crime Drama', 'Action Sci-Fi Thriller', 'Adventure Drama Sci-Fi',
        'Action Adventure Sci-Fi', 'Action Adventure Sci-Fi', 'Action Sci-Fi',
        'Action Thriller', 'Animation Comedy Family', 'Animation Adventure Comedy',
        'Animation Adventure Drama', 'Crime Drama', 'Crime Drama', 'Drama Romance',
        'Drama', 'Drama Thriller', 'Crime Drama', 'Crime Drama Thriller',
        'Action Thriller', 'Action Thriller', 'Action Thriller'
    ],
    'director': [
        'Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan', 'Joss Whedon',
        'Jon Favreau', 'Wachowski', 'Chad Stahelski', 'John Lasseter', 'Andrew Stanton',
        'Roger Allers', 'Quentin Tarantino', 'Francis Ford Coppola', 'Robert Zemeckis',
        'Frank Darabont', 'David Fincher', 'Martin Scorsese', 'Martin Scorsese',
        'Martin Campbell', 'Sam Mendes', 'Christopher McQuarrie'
    ],
    'year': [2008, 2010, 2014, 2012, 2008, 1999, 2014, 1995, 2003, 1994,
             1994, 1972, 1994, 1994, 1999, 1990, 2006, 2006, 2012, 2015],
    'rating': [9.0, 8.8, 8.6, 8.0, 7.9, 8.7, 7.4, 8.3, 8.1, 8.5,
               8.9, 9.2, 8.8, 9.3, 8.8, 8.7, 8.5, 8.0, 7.7, 7.4]
}
df = pd.DataFrame(movies_data)

# --- STEP 2: FEATURE EXTRACTION (The "Brain" of the model) ---
# Combine text features
df['combined_features'] = df['genre'] + ' ' + df['director']

# A. Convert Text to Numbers (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# B. Normalize Numerical Features (Year and Rating)
scaler = MinMaxScaler()
numerical_features = scaler.fit_transform(df[['year', 'rating']])

# C. Combine everything into one matrix and convert to CSR format
# This is where we fixed the NameError and the Subscriptable error!
feature_matrix = hstack([tfidf_matrix, numerical_features]).tocsr()

# --- STEP 3: THE MODEL ---
knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_model.fit(feature_matrix)

# --- STEP 4: RECOMMENDATION FUNCTION ---
def get_movie_recommendations(movie_title, n=5):
    try:
        idx = df[df['title'] == movie_title].index[0]
        movie_vec = feature_matrix[idx]
        
        # Find similar movies
        distances, indices = knn_model.kneighbors(movie_vec.reshape(1, -1), n_neighbors=n+1)
        
        # Format results
        movie_indices = indices[0][1:]
        recs = df.iloc[movie_indices][['title', 'genre', 'director', 'year', 'rating']].copy()
        recs['similarity'] = 1 - distances[0][1:]
        return recs
    except IndexError:
        return None

# --- STEP 5: TEST ---
test_movie = 'Inception'
print(f"\n🎬 IF YOU LIKED: {test_movie}")
results = get_movie_recommendations(test_movie)
if results is not None:
    print(results[['title', 'similarity']])