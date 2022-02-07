
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(1)


def combined_features(row):
    """ Combine the features director, first 4 cast, and title from one row into a string.

    :param row: A row of a DataFrame
    :return: A string of the combined features.
    """
    cast4 = row['cast'].split(', ')[:4]
    cast4 = ' '.join([''.join(x.split()) for x in cast4])

    director = row['director'].replace(' ', '')

    return ' '.join([director, cast4, row['title']])


def add_columns(movies):
    """ Add columns we will use for calculations to the DataFrame

    :param movies: DataFrame of movies & info
    """
    movies['index'] = range(len(movies))
    movies = movies.reset_index()

    movie_features = ["director", "cast", "title"]

    for feature in movie_features:
        movies[feature] = movies[feature].fillna('')

    movies["combined_features"] = movies.apply(combined_features, axis=1)
    return movies


def base_scores(movies):
    """ Compute a similarity matrix for movies, based on director/cast/title

    :param movies: DataFrame of movies & info
    :return: A similarity matrix
    """
    movie_cv = CountVectorizer(stop_words='english')
    count_matrix = movie_cv.fit_transform(movies["combined_features"])
    # print("Count Matrix:", CC_count_matrix.toarray())
    cosine_sim = cosine_similarity(count_matrix)
    # print(cosine_sim)
    return cosine_sim


def fetch_title(movies, title):
    """ Fetch a movie index using the title

    :param movies: DataFrame of movies & info
    :param title: Title of a movie
    :return: The index of the first movie matching the title
    """
    return movies[movies.title == title]["index"].values[0]


def fetch_index(movies, index):
    """ Fetch a movie title using the index

    :param movies: DataFrame of movies & info
    :param index: Index of a movie/row in the DataFrame
    :return: The title of the movie at the given index
    """
    return movies[movies.index == index]["title"].values[0]


def genre_scores(movies):
    """ Compute a similarity matrix for movies by genre

    :param movies: DataFrame of movies & info
    :return: A similarity matrix
    """
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(movies['genres'])
    genre_sims = cosine_similarity(count_matrix)
    return genre_sims


def description_scores(movies):
    """ Compute a similarity matrix for movies by description

    :param movies: DataFrame of movies & info
    :return: A similarity matrix
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies["description"])
    description_sims = cosine_similarity(tfidf_matrix)
    return description_sims


def year_scores(movies):
    """ Compute a difference matrix for movies by year
    Note: Unlike the other similarity matrices, 0 is most similar

    :param movies: DataFrame of movies & info
    :return: A difference matrix
    """
    y1 = movies['year'].values
    y2 = movies['year'].values
    year_diffs = np.abs((y1[None, :] - y2[:, None]) / 100)
    return year_diffs


def norm(data, ceiling=float('inf'), floor=float('-inf')):
    """ Normalize an array of data with the provided ceiling a floor.

    :param data: An array of data
    :param ceiling: Any value >= ceiling becomes 1
    :param floor: Any value <= floor becomes 0
    :return:
    """

    if ceiling < float('inf'):
        ceiling = max(data)

    if floor > float('-inf'):
        floor = min(data)

    result = ((data - floor) / (ceiling - floor))
    for x in range(len(result)):
        if result[x] > 1:
            result[x] = 1
        if result[x] < 0:
            result[x] = 0

    return result


def combine(base_scores, genre_scores, description_scores, year_scores, popularity_vector, rating_vector):
    """ Combines similarity matrices and vectors using addition and weights

    :param base_scores: Base similarity matrix
    :param genre_scores: Genre similarity matrix
    :param description_scores: Description similarity matrix
    :param year_scores: Year difference matrix
    :param popularity_vector: Popularity vector
    :param rating_vector: Rating vector
    :return: Resulting similarity matrix from the weighted computation
    """
    weighted_base = np.multiply(base_scores, 2.0)
    weighted_genres = np.multiply(genre_scores, 2.5)
    weighted_descriptions = np.multiply(description_scores, 1.5)
    weighted_years = np.multiply(year_scores, -4)

    weighted_pop = np.multiply(popularity_vector, 1.5)
    weighted_ratings = np.multiply(rating_vector, 3)

    combined_matrix = weighted_base + weighted_genres + weighted_descriptions + weighted_years
    combined_matrix = combined_matrix + (weighted_pop + weighted_ratings)[:, np.newaxis]

    return combined_matrix


def create_similarity_matrix(movies_df):
    movies_df = add_columns(movies_df)
    base_m = base_scores(movies_df)
    year_m = year_scores(movies_df)
    genre_m = genre_scores(movies_df)
    description_m = description_scores(movies_df)
    pop_v = norm(movies_df['popularity'], ceiling=50, floor=0)
    rating_v = norm(movies_df['average_vote'], ceiling=10, floor=0)
    scores = combine(base_m, genre_m, description_m, year_m, pop_v, rating_v)
    return scores
