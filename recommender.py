import numpy as np


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


"""# Make the Recommendations
Provide a ranked list of the 'most similar' movies to the input movies provided.
"""


def recommend(movies, scores, titles, count):
    """ Make a recommendation list based on a list of titles

    :param movies: DataFrame of movies & info
    :param scores: Similarity matrix for the movies
    :param titles: List of input titles
    :param count: Number of recommendations to output
    :return:
    """
    proximity_arr = np.zeros([len(movies), 2])
    proximity_arr[:,0] = np.array(movies["index"], dtype=int)
    for title in titles:
        movie_index = fetch_title(movies, title)
        proximity_row = np.array(scores[movie_index])
        proximity_arr[:,1] = proximity_arr[:,1] + proximity_row

    movie_proximity_sorted = proximity_arr[proximity_arr[:, 1].argsort()][::-1]

    i = 0
    n = 1
    print("Recommendations:")
    while n <= count:
        title = fetch_index(movies, int(movie_proximity_sorted[i, 0]))
        if title not in titles:
            print(f"{n}. {title}")
            n += 1
        i += 1
