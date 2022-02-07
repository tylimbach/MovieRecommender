import recommender
import similarity
import pandas as pd


def main():
    # read in the data
    movies_df = pd.read_csv("cleaned_movies.csv")

    # calculate similarity
    scores = similarity.create_similarity_matrix(movies_df)

    # menu loop
    loop(movies_df, scores, 10)


def loop(movies, scores, length):
    count = input("How many movies would you like to enter? (Enter Q to quit): ")
    while count != "Q":
        titles = []
        for i in range(int(count)):
            titles.append(input("Enter a title: "))

        recommender.recommend(movies, scores, titles, length)
        count = input("How many movies would you like to enter? (Enter Q to quit): ")


if __name__ == '__main__':
    main()
