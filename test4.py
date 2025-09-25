import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import requests
import time

def fetch_poster(movie_id):
    api_key = "66357c6b8b9e5449a6ed31861ece47ed"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    for _ in range(5):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
                return full_path
            else:
                break
        except requests.exceptions.RequestException:
            time.sleep(0.5)
    return "https://via.placeholder.com/500x750.png?text=No+Poster+Available"

@st.cache_data
def load_and_process_data():
    movies_df = pd.read_csv('tmdb_5000_movies.csv')
    credits_df = pd.read_csv('tmdb_5000_credits.csv')
    movies_df = movies_df.merge(credits_df, on="title")
    movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies_df.dropna(inplace=True)

    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    movies_df['genres'] = movies_df['genres'].apply(convert)
    movies_df['keywords'] = movies_df['keywords'].apply(convert)

    def convert2(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L
    movies_df['cast'] = movies_df['cast'].apply(convert2)

    def director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L
    movies_df['crew'] = movies_df['crew'].apply(director)
    movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())
    movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']
    
    new_df = movies_df[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    stop_words_nltk = set(stopwords.words('english'))
    new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in stop_words_nltk]
    ))
    
    cv = CountVectorizer(max_features=10000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity_matrix = cosine_similarity(vectors)
    
    return new_df, similarity_matrix

movies, similarity = load_and_process_data()

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    input_movie_tags = set(movies.iloc[movie_index].tags.split())
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    
    recommended_movies_data = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        rec_index = i[0]
        rec_title = movies.iloc[rec_index].title
        rec_tags = set(movies.iloc[rec_index].tags.split())
        
        common_tags = list(input_movie_tags.intersection(rec_tags))
        
        recommended_movies_data.append({
            "title": rec_title,
            "poster_url": fetch_poster(movie_id),
            "common_tags": common_tags 
        })
        time.sleep(0.1)
        
    return recommended_movies_data

st.set_page_config(layout="wide")
st.title('Movie Recommendation System')

selected_movie_name = st.selectbox(
    'Select a movie you like, and we will recommend 10 similar ones:',
    movies['title'].values)

if st.button('Recommend'):
    with st.spinner('Finding recommendations...'):
        recommendations = recommend(selected_movie_name)
    
    st.write("Here are your recommendations:")
    
    cols = st.columns(5)
    for index, movie in enumerate(recommendations[:5]):
        with cols[index]:
            st.text(movie['title'])
            st.image(movie['poster_url'])
            if movie['common_tags']:
                tags_str = ", ".join(movie['common_tags'][:3])
                st.caption(f"Tags: {tags_str}")
            
    cols = st.columns(5)
    for index, movie in enumerate(recommendations[5:]):
        with cols[index]:
            st.text(movie['title'])
            st.image(movie['poster_url'])
            if movie['common_tags']:
                tags_str = ", ".join(movie['common_tags'][:3])
                st.caption(f"Tags: {tags_str}")