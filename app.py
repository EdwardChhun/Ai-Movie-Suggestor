# Importing the libraries needed
import pandas as pd  # for working with data
from flask import Flask, render_template, request  # for the web app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Reading the movie files
movies = pd.read_csv('tmdb_5000_movies.csv')  # This file has info about movies
movies['overview'] = movies['overview'].fillna('')

# Prepare to understand movies' descriptions using scikit-learn
tfidf = TfidfVectorizer(stop_words='english')  # This will help us understand movie descriptions
tfidf_matrix = tfidf.fit_transform(movies['overview'])  # This will help us understand movie descriptions

# Let's find out how movies are similar to each other
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)  # This will help us understand movie descriptions

# Make a series to find movies quickly
indices = pd.Series(movies.index, index=movies['title'])


# Make a function that will help us find movies similar to the one we like
def recommend_movies(title, cosine_sim=cosine_sim):
    # If we don't have a movie, say so
    if title not in indices:
        return 'Sorry, we don\'t have that movie. Please try another one.'
    # Find the movie's position in our list
    idx = indices[title]
    # Calculate which movies are most similar
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies from most to least similar
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Find the top 10 most similar movies for the movie we entered
    sim_scores = sim_scores[1:11]
    # Get the names of the movies just now
    movie_titles = [movies['title'].iloc[i] for i, _ in sim_scores]

    # Return the movie names
    return ', '.join(movie_titles)


# Test the function
user_input = input("Type in your desired movie: ")
print(recommend_movies(user_input))
