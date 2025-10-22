import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

nltk.download('punkt_tab')
nltk.download('stopwords')

df = pd.read_csv("IMDB_Top250Engmovies2_OMDB_Detailed.csv")

st.title("Sistema de Recomendação de Filmes")

st.subheader("Planilha com 250 filmes")

st.dataframe(df)



#Tratamento de Dados
df['clean_plot'] = df['Plot'].str.lower()
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('\s+', ' ', x))
df['clean_plot'] = df['clean_plot'].apply(lambda x: nltk.word_tokenize(x))
df['Genre'] = df['Genre'].apply(lambda x: x.split(','))
df['Actors'] = df['Actors'].apply(lambda x: x.split(',')[:3])
df['Director'] = df['Director'].apply(lambda x: x.split(','))


#retirando conjunções em inglês
stop_words = nltk.corpus.stopwords.words('english')
plot = []
for sentence in df['clean_plot']:
    temp = []
    for word in sentence:
        if word not in stop_words and len(word) >= 3:
            temp.append(word)
    plot.append(temp)

df['clean_plot'] = plot

def clean(sentence):
    temp = []
    for word in sentence:
        temp.append(word.lower().replace(' ', ''))
    return temp

df['Genre'] = [clean(x) for x in df['Genre']]
df['Actors'] = [clean(x) for x in df['Actors']]
df['Director'] = [clean(x) for x in df['Director']]

columns = ['clean_plot', 'Genre', 'Actors', 'Director']
l = []
for i in range(len(df)):
    words = ''
    for col in columns:
        words += ' '.join(df[col][i]) + ' '
    l.append(words)

df['clean_input'] = l
df = df[['Title', 'clean_input']]

tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df['clean_input'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(features, features)

index = pd.Series(df['Title'])

def recommend_movies(title):
    movies = []
    idx = index[index == title].index[0]
    score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top10 = list(score.iloc[1:11].index)
    
    for i in top10:
        movies.append(df['Title'][i])
    return movies

#Interface do Streamlit

filme_input = st.text_input("Digite o nome de um filme para obter recomendações:")

if st.button('Recomendar Filmes'):
    if filme_input:
        try:
            recommended_list = recommend_movies(filme_input)
            st.subheader(f"Filmes recomendados com base em '{filme_input}':")
            for rank, title in enumerate(recommended_list, 1):
                st.write(f"{rank}. {title}")
        except IndexError:
            st.error(f'O filme "{filme_input}" não foi encontrado. Tente um filme válido.')
    else:
        st.warning("Por favor, digite o nome de um filme.")

