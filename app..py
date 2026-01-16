import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

# --- PAGE CONFIG ---
st.set_page_config(page_title="Movie Matcher", page_icon="🎬")

# --- STEP 1: THE DATASET ---
@st.cache_data # This keeps the app fast by saving the data in memory
def load_data():
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
    return pd.DataFrame(movies_data)

df = load_data()

# --- STEP 2: FEATURE EXTRACTION ---
df['combined_features'] = df['genre'] + ' ' + df['director']

# Text Features (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Numerical Features (Year/Rating)
scaler = MinMaxScaler()
num_features = scaler.fit_transform(df[['year', 'rating']])

# Combine and convert to CSR for row-access
feature_matrix = hstack([tfidf_matrix, num_features]).tocsr()

# --- STEP 3: TRAIN KNN ---
model = NearestNeighbors(n_neighbors=6, metric='cosine')
model.fit(feature_matrix)

# --- STEP 4: STREAMLIT UI ---
st.title("🎬 Movie Recommendation System")
st.write("Pick a movie you love, and I'll find 5 similar ones using KNN.")

selected_movie = st.selectbox("Select a movie from the database:", df['title'].values)

if st.button('Show Recommendations'):
    # Get Index
    idx = df[df['title'] == selected_movie].index[0]
    
    # Calculate Neighbors
    distances, indices = model.kneighbors(feature_matrix[idx].reshape(1, -1))
    
    # Display Results
    st.subheader(f"Because you liked {selected_movie}:")
    
    # We skip the first result [0] because it's the movie itself
    cols = st.columns(5)
    for i in range(1, 6):
        with cols[i-1]:
            movie_idx = indices[0][i]
            st.write(f"**{df.iloc[movie_idx]['title']}**")
            st.caption(f"⭐ Rating: {df.iloc[movie_idx]['rating']}")
            st.caption(f"📅 {df.iloc[movie_idx]['year']}")
            similarity = (1 - distances[0][i]) * 100
            st.info(f"{similarity:.1f}% Match")

st.markdown("---")
st.caption("Built with ❤️ using Python, Scikit-learn, and Streamlit")
import requests

def get_poster(movie_title):
    api_key = "211d2143d5d730a79006ba1f37c3c305"
    # 1. Search for the movie ID
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    response = requests.get(url).json()
    
    # 2. Grab the poster path
    if response['results']:
        path = response['results'][0]['poster_path']
        return f"https://image.tmdb.org/t/p/w500/{path}"
    return "https://via.placeholder.com/500x750?text=No+Poster+Found"