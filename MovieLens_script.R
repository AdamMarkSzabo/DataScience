if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


## Capstone MovieLens Assaignment

#Departing data set

mytest_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-mytest_index,]
test_set <- edx[mytest_index,]


## RMSE function

RMSE <- function(actual_rating, predicted_rating) {
  sqrt(mean((actual_rating - predicted_rating)^2))
}


## Baseline, average approach

mu_hat <- mean(train_set$rating)
rmse_mu <- RMSE(test_set$rating, mu_hat)

rmse_resTable <- tibble(method = "Predicting by average", RMSE = rmse_mu)


## Movies effect on prediction

movie_effect <- train_set %>% group_by(movieId) %>% summarise(b_i = mean(rating - mu_hat))

qplot(b_i,data = movie_effect, bins = 20)

movie_effect %>% summarise(sum = sum(b_i))

pred_movies <- test_set %>% group_by(movieId) %>% left_join(movie_effect, "movieId") %>% pull(b_i)
sum(is.na(pred_movies)) 
pred_movies[is.na(pred_movies)] <- mu_hat # NA removal
pred_mui <- mu_hat + pred_movies

# RMSE test for movies impact
rmse_mui <- RMSE(test_set$rating, pred_mui)
rmse_resTable <- rmse_resTable %>% add_row(method = "Movie effect on prediction", RMSE =  rmse_mui)


## Users effect on prediction

# user effect alone
user_effect <- train_set %>% group_by(userId) %>% summarise(b_u =  mean(rating) - mu_hat)
pred_usr <- test_set %>% 
  group_by(movieId) %>% 
  left_join(user_effect, "userId") %>%
  mutate(prediction = mu_hat + b_u) %>%
  pull(prediction)

rmse_usr <- RMSE(test_set$rating,pred_usr)

rmse_mov_usr <- RMSE(test_set$rating,pred_usr)
rmse_resTable <- rmse_resTable %>% add_row(method = "Users effect on prediction", RMSE =  rmse_usr)


## Genres effect on the prediction

genres_table <- train_set %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating)) %>%
  filter(n >= 500) %>% 
  mutate(genres = reorder(genres, avg))

str(genres_table)

genres_table %>% arrange(desc(avg)) %>%
  head(.,20) %>%
  ggplot(aes(x = genres, y = avg)) + 
  geom_point()+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

genres_table %>% arrange(desc(avg)) %>%
  tail(.,20) %>%
  ggplot(aes(x = genres, y = avg)) + 
  geom_point()+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


genres_effect <- train_set %>%  
  group_by(genres) %>% 
  summarise(b_g = mean(mean(rating) - mu_hat))

# prediction model
pred_genres <- test_set %>% 
  group_by(movieId) %>%
  left_join(genres_effect, "genres") %>% 
  mutate(prediction = mu_hat + b_g) %>%
  pull(prediction)
if(sum(is.na(pred_genres))) pred_genres[is.na(pred_genres)] <- mu_hat

rmse_gen <- RMSE(test_set$rating, pred_genres)
rmse_resTable <- rmse_resTable %>% add_row(method = "Genres effect on prediction", RMSE =  rmse_gen)

# genre + user combined
genres_effect_bad <- train_set %>%  
  left_join(user_effect, "userId") %>% 
  group_by(genres) %>% 
  summarise(b_g = mean(mean(rating) - mu_hat - b_u))  
# user related to genres not as punctual but very close to checking only the genres

# genre effect alone
pred_genres2 <- test_set %>% 
  group_by(movieId) %>%
  left_join(genres_effect_bad, "genres") %>% 
  mutate(prediction = mu_hat + b_g) %>%
  pull(prediction)
if(sum(is.na(pred_genres2))) pred_genres2[is.na(pred_genres2)] <- mu_hat

rmse_gen2 <- RMSE(test_set$rating,pred_genres2)
rmse_gen2


## Combining effects for the model

# Movie + user + genre 
pred_mov_usr_gen <- test_set %>% 
  group_by(movieId) %>% 
  left_join(movie_effect, "movieId") %>%
  left_join(user_effect, "userId") %>%
  left_join(genres_effect, "genres") %>%
  mutate(prediction = mu_hat + b_i + b_u + b_g) %>%
  pull(prediction)
if(sum(is.na(pred_mov_usr_gen))) pred_mov_usr_gen[is.na(pred_mov_usr_gen)] <- mu_hat

rmse_gen <- RMSE(test_set$rating,pred_genres)
rmse_gen
# not very good rmse


# Movie and user effect combined
pred_mov_usr <- test_set %>% 
  group_by(movieId) %>% 
  left_join(movie_effect, "movieId") %>%
  left_join(user_effect, "userId") %>%
  mutate(prediction = mu_hat + b_i + b_u) %>%
  pull(prediction)
if(sum(is.na(pred_mov_usr))) pred_mov_usr[is.na(pred_mov_usr)] <- mu_hat

rmse_mov_usr <- RMSE(test_set$rating, pred_mov_usr)
rmse_mov_usr

rmse_resTable <- rmse_resTable %>% add_row(method = "Movie + User effect on prediction", RMSE =  rmse_mov_usr)

## Regularization

lambda <- seq(0, 10, 0.25)

rmse_reg <- sapply(lambda, function(lambda){
  
  movies_reg <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu_hat)/(n()+lambda))
  
  users_reg <- train_set %>% 
    left_join(movies_reg, by="movieId") %>%
    group_by(userId) %>% 
    summarise(b_u = sum(rating - b_i - mu_hat)/(n()+lambda))
  
  pred_mov_usr_reg <- test_set %>% 
    group_by(movieId) %>% 
    left_join(movies_reg, "movieId") %>%
    left_join(users_reg, "userId") %>%
    mutate(prediction = mu_hat + b_i + b_u) %>%
    pull(prediction)
  if(sum(is.na(pred_mov_usr_reg))) pred_mov_usr_reg[is.na(pred_mov_usr_reg)] <- mu_hat
  
  return(RMSE(test_set$rating, pred_mov_usr_reg ))
})

qplot(lambda, rmse_reg)

lambda[which.min(rmse_reg)]

# RMSE with 4.75 lambda
rmse_reg[which.min(rmse_reg)]

rmse_resTable <- rmse_resTable %>% add_row(method = "Regularized Movie + User effect on prediction", RMSE =  rmse_reg[which.min(rmse_reg)])


# Final model is used on the validation set to see how it performs.
# Final Root mean square error value is defined by: 
  

l <- 4.75

movies_reg <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu_hat)/(n()+l))

users_reg <- edx %>% 
  left_join(movies_reg, by="movieId") %>%
  group_by(userId) %>% 
  summarise(b_u = sum(rating - b_i - mu_hat)/(n()+l))

pred_mov_usr_reg <- validation %>% 
  group_by(movieId) %>% 
  left_join(movies_reg, "movieId") %>%
  left_join(users_reg, "userId") %>%
  mutate(prediction = mu_hat + b_i + b_u) %>%
  pull(prediction)
if(sum(is.na(pred_mov_usr_reg))) pred_mov_usr_reg[is.na(pred_mov_usr_reg)] <- mu_hat

RMSE(validation$rating, pred_mov_usr_reg)
