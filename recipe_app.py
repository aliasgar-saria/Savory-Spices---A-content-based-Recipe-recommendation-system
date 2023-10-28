import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load the dataset
file_path = 'cleaned.csv'
data = pd.read_csv(file_path, header=None)

# Text Preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = str(text).lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-z\s]', '', text)
    return text

# Preprocess the 'Description', 'Ingredients Name', and 'Instructions' columns
data[2] = data[2].apply(preprocess_text)
data[6] = data[6].astype(str).apply(preprocess_text)
data[10] = data[10].apply(preprocess_text)

# Combine 'Description', 'Ingredients Name', and 'Instructions' into a single text column
data['Combined Text'] = data[2] + ' ' + data[6] + ' ' + data[10]

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the combined text data into a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Combined Text'])

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recipe recommendations based on cosine similarity
def get_recommendations(recipe_index, top_n=5):
    # Get the cosine similarity scores for the given recipe
    sim_scores = list(enumerate(cosine_sim_matrix[recipe_index]))
    
    # Sort the recipes based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the top N similar recipes
    sim_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    # Get the recipe names and images for the top N similar recipes
    recommended_recipes = data.iloc[sim_indices, [1, 11, 2]]
    
    return recommended_recipes

# Streamlit App
st.title('Savory Spices - Recipe Recommendation System')

# User input: Select a recipe from the dropdown list
selected_recipe = st.selectbox('Select a recipe:', data[1])

# Get the index of the selected recipe
recipe_index = data[data[1] == selected_recipe].index[0]

# Get recommendations for the selected recipe
recommended_recipes = get_recommendations(recipe_index, top_n=5)

# Display recommendations
st.subheader('Recommended Recipes:')
for index, row in recommended_recipes.iterrows():
    st.image(row[11], width=200)
    st.text(row[1])
    st.text(row[2])
    st.write('---')
