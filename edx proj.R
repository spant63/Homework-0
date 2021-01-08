
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

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

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
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


#Split the edx set into Training and Test sets#
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
training <- edx[-test_index,]
temp <- edx[test_index,]

#Add movieId, userId, and genres are in the training set#
testing <- temp %>% 
  semi_join(training, by = "movieId") %>%
  semi_join(training, by = "userId")


# Add rows removed from test set back into train set
removed <- anti_join(temp, testing)
training <- rbind(training, removed)

rm(test_index, temp, removed)

#summary of the data
head(edx)
summary(edx)

#summary of the variables being used in the model
#Movie Summary
edx %>% group_by(movieId) %>% summarize(n = n())
#User Summary
edx %>% group_by(userId) %>% summarize(n = n())
#Genre Summary
edx %>% group_by(genres) %>% summarize(n = n()) 


#Initial Model#
avg <- mean(training$rating)
avg

#Movie Effect#
movieEffect <- training %>% 
  group_by(movieId) %>% 
  summarize(m_e = mean(rating - avg))


predictions <- avg + testing %>% 
  left_join(movieEffect, by = "movieId") %>% 
  pull(m_e)

rmse_movie_effect <- RMSE(testing$rating, predictions)
rmse_movie_effect


#Movie + User Effect#
userEffect <- training %>%
  left_join(movieEffect, by = 'movieId') %>%
  group_by(userId) %>% 
  summarize(u_e = mean(rating - avg - m_e))

predictions_user <- testing %>% 
  left_join(movieEffect, by = "movieId") %>%
  left_join(userEffect, by = 'userId') %>%
  mutate(predict = avg + m_e + u_e) %>%
  pull(predict)

rmse_user_effect <- RMSE(testing$rating, predictions_user)
rmse_user_effect


#Movie + User + Genre Effect#
genreEffect <- training %>%
  left_join(movieEffect, by = 'movieId') %>%
  left_join(userEffect, by='userId') %>%
  group_by(genres) %>% 
  summarize(g_e = mean(rating - avg - m_e - u_e))

predictions_genres <- testing %>% 
  left_join(movieEffect, by = "movieId") %>%
  left_join(userEffect, by = 'userId') %>%
  left_join(genreEffect, by ='genres') %>%
  mutate(predict = avg + m_e + u_e + g_e) %>%
  pull(predict)

rmse_genre_effect <- RMSE(testing$rating, predictions_genres)
rmse_genre_effect


#Regularization w/ Test & Train set#

avg <- mean(training$rating)

lambda <- seq(0, 10, 0.5)

rmse <- sapply(lambda, function(lmd){
  

  m_e <- training %>% 
    group_by(movieId) %>%
    summarize(m_e = sum(rating - avg)/(n()+lmd))
  
  u_e <- training %>% 
    left_join(m_e, by="movieId") %>%
    group_by(userId) %>%
    summarize(u_e = sum(rating - m_e - avg)/(n()+lmd))
  
  g_e <- training %>%
    left_join(m_e, by="movieId") %>%
    left_join(u_e, by="userId") %>%
    group_by(genres) %>%
    summarize(g_e = sum(rating - m_e - u_e - avg)/(n()+lmd))

    
  predicted_ratings <-  testing %>% 
    left_join(m_e, by = "movieId") %>%
    left_join(u_e, by = "userId") %>%
    left_join(g_e, by = 'genres') %>%
    mutate(pred = avg + m_e + u_e + g_e) %>%
    pull(pred)
  
  RMSE(predicted_ratings, testing$rating)
})

#RMSE Value#
lowest_rmse <- rmse[which.min(rmse)]
lowest_rmse

#Lambda Value#
lowest_lambda <- lambda[which.min(rmse)]
lowest_lambda


#Visualization of lowest point in lambda vs rmse
qplot(lowest_lambda, lowest_rmse)  




#Final prediction w/ Validation set#

avg <- mean(edx$rating)

lambda_Validation <- seq(0, 10, 0.5)

rmse_Validation <- sapply(lambda_Validation, function(lmd){
  

  
  m_e <- edx %>% 
    group_by(movieId) %>%
    summarize(m_e = sum(rating - avg)/(n()+lmd))
  
  u_e <- edx %>% 
    left_join(m_e, by="movieId") %>%
    group_by(userId) %>%
    summarize(u_e = sum(rating - m_e - avg)/(n()+lmd))
  
  g_e <- edx %>%
    left_join(m_e, by="movieId") %>%
    left_join(u_e, by="userId") %>%
    group_by(genres) %>%
    summarize(g_e = sum(rating - m_e - u_e - avg)/(n()+lmd))
  
  
  predicted_ratings <- 
    validation %>% 
    left_join(m_e, by = "movieId") %>%
    left_join(u_e, by = "userId") %>%
    left_join(g_e, by = 'genres') %>%
    mutate(pred = avg + m_e + u_e + g_e) %>%
    pull(pred)
  
  RMSE(predicted_ratings, validation$rating)
})

#RMSE Value#
lowest_rmses_validation_final <- rmse_Validation[which.min(rmse_Validation)]
lowest_rmses_validation_final

#Lambda Value#
lowest_lambda_validation_final<- lambda_Validation[which.min(rmse_Validation)]
lowest_lambda_validation_final


#Visualization of lowest point in lambda vs rmse
qplot(lowest_lambda_validation_final, lowest_rmses_validation_final)  






  

  
  
