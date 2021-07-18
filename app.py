import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


app = Flask(__name__)
# from tensorflow.keras.models import load_model
import os
import numpy as np
import pandas as pd

df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')
df1.columns = ['id', 'tittle', 'cast', 'crew']
df2 = df2.merge(df1, on='id')
df2.head()

m = df2['vote_count'].quantile(0.9)


q_movies = df2.copy().loc[df2['vote_count'] >= m]


C = df2['vote_average'].mean()



def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('score', ascending=False)
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)

from sklearn.feature_extraction.text import TfidfVectorizer

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])



from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk

nltk.download('punkt')
nltk.download('stopwords')


def get_sim_movies(X, Y):
    z = Y

    X = X.lower()
    Y = Y.lower()
    X_list = word_tokenize(X)
    Y_list = word_tokenize(Y)

    sw = stopwords.words('english')
    l1 = [];
    l2 = []

    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}

    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
    if cosine > 0.4:
        return z
    else:
        return -1





@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact.html')
def mod():
    return render_template('contact.html')

@app.route('/index.html')
def ind():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    # if(int_features[5]=='yes'):
    #     int_features[5] = 1
    # else:
    #     int_features[5] = 0
    # if (int_features[6] == 'yes'):
    #     int_features[6] = 1
    # else:
    #     int_features[6] = 0
    # if (int_features[7] == 'yes'):
    #     int_features[7] = 1
    # else:
    #     int_features[7] = 0
    #
    # int_features_new = [int(x) for x in int_features]
    # final_features = [np.array(int_features_new)]
    # u = pd.DataFrame(final_features, columns=['profile pic', 'fullname words', 'name==username', 'description length', 'private',
    #                              '#posts', '#followers', '#follow'])
    #
    # prediction = model.predict_classes(u)

    x = int_features[0]

    j = []
    for i in q_movies['title']:

        try:
            a = get_sim_movies(x, i)
            if a != -1:
                j.append(a)
        except:
            continue


    a=[]
    for i in j:
        a.append(get_recommendations(i))
    Fin=[]
    for i in a:
        Val=i.values
        Fin.append(Val)
    # print(a.values)
    return render_template('contact.html', prediction_text = '{}'.format(Fin))

if __name__ == "__main__":
    app.run(debug=True)