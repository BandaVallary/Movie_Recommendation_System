# Movie Recommendation System

![Image of movie clapperboard](images/movie_img_cropped.jpeg)

## Summary

In this project, we used the MovieLens dataset, which includes 100,836 user ratings from 610 users, averaging 3.5 out of 5. This dataset is ideal for building a recommendation system, leveraging both collaborative and content-based filtering.

Data preparation involved handling missing values and formatting for analysis using Pandas, while Scikit-Learn and Keras were used for preprocessing. For visualization, we utilized Seaborn and Matplotlib.

For modeling, we applied Surprise for collaborative filtering and explored various metrics. Additionally, we built a neural network with TensorFlow and Keras, incorporating embedding layers and techniques to prevent overfitting.

Model validation was conducted using cross-validation, and the final model achieved a Root Mean Squared Error (RMSE) of 0.869, indicating effective personalized movie suggestions. We employed an 80-20 train-test split for performance validation.

## Overview

This project aims to develop a recommendation system that provides personalized movie recommendations based on user ratings. Utilizing the [MovieLens dataset](https://web.archive.org/web/20240828133414/https://grouplens.org/datasets/movielens/latest/) from the GroupLens research lab at the University of Minnesota, the model will be trained on a subset of the dataset containing 100,000 user ratings.

This system can be valuable for streaming platforms and movie enthusiasts, offering tailored movie suggestions to enhance user experience and engagement.The project will involve several steps, including data cleaning, exploratory data analysis, feature engineering, model selection, and evaluation.

Throughout this project, we will also explore the relationships between different variables and their impact on movie recommendations. This will help us gain insights into user preferences and identify potential areas for improvement. Overall, this project has the potential to provide valuable insights and practical applications for the entertainment industry. By developing a recommendation system that can accurately suggest movies, streaming platforms can better engage their users, improve customer satisfaction, and increase viewership.

## Business Understanding

The entertainment industry, especially streaming platforms, is fiercely competitive, with a constant need to enhance user engagement and satisfaction. A key challenge is delivering personalized content recommendations to reduce churn rates.

Personalized recommendations can significantly boost user engagement and retention. Therefore, a robust recommendation system is essential for suggesting movies based on user preferences. By providing tailored movie suggestions, streaming platforms can create a better viewing experience.

The project's business value lies in improving content recommendation strategies, increasing user satisfaction, and reducing churn. An effective recommendation system can enhance user engagement, leading to higher viewership and subscription renewals, thus providing a competitive edge and driving revenue growth.

## Data Understanding

The data used in this project is the [MovieLens dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) from the GroupLens research lab, containing 100,000 user ratings. It includes three main files:

- `movies.csv`: Contains movie details such as unique identifiers, titles (with release years), and genres.

- `ratings.csv`: Captures user ratings, including user IDs, movie IDs, rating values (1 to 5), and timestamps.

- `tags.csv`: Records user-generated tags for movies, featuring user IDs, movie IDs, tags, and timestamps.

The `movieId` is consistent across all files, while `userId` is consistent between the ratings and tags files. This structure facilitates effective analysis for the recommendation system.
