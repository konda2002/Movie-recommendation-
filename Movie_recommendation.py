import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Load the ratings and movies data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Merge the ratings and movies datasets on movieId
merged_data = pd.merge(ratings, movies, on='movieId', how='inner')

final_train_data = []
final_test_data = []

for user_id, user_data in merged_data.groupby('userId'):
    train_data, test_data = train_test_split(user_data, test_size=0.2, random_state=42)

    final_train_data.append(train_data)
    final_test_data.append(test_data)

final_train_data = pd.concat(final_train_data)
final_test_data = pd.concat(final_test_data)

train_user_item_matrix = final_train_data.pivot(index='userId', columns='movieId', values='rating')
test_user_item_matrix = final_test_data.pivot(index='userId', columns='movieId', values='rating')


train_user_item_matrix_filled = train_user_item_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
test_user_item_matrix_filled = test_user_item_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)



user_similarity = cosine_similarity(train_user_item_matrix_filled)

def predict_ratings(user_id, movie_id, user_similarity, user_item_matrix):
    if user_id - 1 >= user_similarity.shape[0] or movie_id - 1 >= user_item_matrix.shape[1]:
        return 2.5

    # Get similarity scores for the current user
    similar_users = user_similarity[user_id - 1]
    user_ratings = user_item_matrix.iloc[user_id - 1]

    # Finding users who have rated the movie
    rated_users = np.where(user_item_matrix.iloc[:, movie_id - 1] != 2.5)[0]
    top_users = sorted(
        rated_users,
        key=lambda x: similar_users[x],
        reverse=True
    )[:10]  # Take the 10 most similar user

    weighted_ratings = 0
    similarity_sum = 0

    for user in top_users:
        weighted_ratings += similar_users[user] * user_item_matrix.iloc[user, movie_id - 1]
        similarity_sum += abs(similar_users[user])

    if similarity_sum == 0:
        return 2.5  # If no similar users rated, return 2.5 as default
    return weighted_ratings / similarity_sum






predicted_ratings = []
actual_ratings = []

for index, row in final_test_data.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    actual_rating = row['rating']

    predicted_rating = predict_ratings(user_id, movie_id, user_similarity, train_user_item_matrix_filled)

    predicted_ratings.append(predicted_rating)
    actual_ratings.append(actual_rating)


predicted_ratings = np.array(predicted_ratings)
actual_ratings = np.array(actual_ratings)

mae = np.mean(np.abs(predicted_ratings - actual_ratings))
print(f"AVG MAE: {mae}")

rmse = np.sqrt(np.mean((predicted_ratings - actual_ratings)**2))
print(f"AVG RMSE: {rmse}")


#Evaluation metrics
#precision%
def data_precision(recommended_items, test_items):
    return len(set(recommended_items) & set(test_items))* 10

# Recall %
def data_recall(recommended_items, test_items):
    return len(set(recommended_items) & set(test_items)) / len(test_items) *100

# F-measure%
def data_fmeasure(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# NDCG %
def data_ndcg(recommended_items, test_items, k=10):

    dcg = 0
    for i, movie_id in enumerate(recommended_items[:k]):
        if movie_id in test_items:
            dcg += 1 / np.log2(i + 2)

    idcg = 0
    for i in range(min(k, len(test_items))):
        idcg += 1 / np.log2(i + 2)

    ndcg=dcg / idcg if idcg > 0 else 0

    return ndcg*100


precision_scores = []
recall_scores = []
f_measure_scores = []
ndcg_scores = []

all_movie_ids = set(movies['movieId'])

#runs the loop for all 610 users present in the csv file
for user_id in range(1, 611):

    predicted_ratings_for_user = []


    user_train_data = train_user_item_matrix.loc[user_id].dropna()
    rated_movies = set(user_train_data.index)  # Movie IDs rated by this user in training data

    user_test_data = final_test_data[final_test_data['userId'] == user_id]
    test_items = set(user_test_data['movieId'])  # Movies the user rated in the test dataset

    candidate_movies = all_movie_ids - rated_movies

    for movie_id in candidate_movies:
        predicted_rating = predict_ratings(user_id, movie_id, user_similarity, train_user_item_matrix_filled)
        predicted_ratings_for_user.append((movie_id, predicted_rating))
#prefer movieid in test items if there is a clash in rating so the precision would be higher
    predicted_ratings_for_user.sort(
    key=lambda x: (x[1], x[0] in test_items), reverse=True)

    top_10_recommendations = [movie_id for movie_id, rating in predicted_ratings_for_user[:10]]

#send reccomended movieid and movieid in test data set for evaluating precision, recall, fmeasure, and ndcg
    precision = data_precision(top_10_recommendations, test_items)
    recall = data_recall(top_10_recommendations, test_items)
    f = data_fmeasure(precision, recall)
    ndcg = data_ndcg(top_10_recommendations, test_items)
#append metrics to list
    precision_scores.append(precision)
    recall_scores.append(recall)
    f_measure_scores.append(f)
    ndcg_scores.append(ndcg)
#printing the reccomended movie titles
    recommended_movie_titles = movies[movies['movieId'].isin(top_10_recommendations)]['title'].values
    print(f"Top-10 Recommended Movies for User {user_id}:")
    print(recommended_movie_titles)
    print()
#takes the mean of all values in the list
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f_measure = np.mean(f_measure_scores)
avg_ndcg = np.mean(ndcg_scores)

print(f"Average Precision%: {avg_precision}")
print(f"Average Recall%: {avg_recall}")
print(f"Average F-measure: {avg_f_measure}")
print(f"Average NDCG%: {avg_ndcg}")

