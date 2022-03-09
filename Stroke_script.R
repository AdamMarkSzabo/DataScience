if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(ROSE)) install.packages("ROSE", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")

library(dslabs)
library(tidyverse)    # includes readr
library(readxl)
library(caret)
library(data.table)
library(randomForest)
#install.packages("ROSE")
library(ROSE)
library(gridExtra)
library(rpart)

# Data set is from Keggle, 
# thanks for author fedesoriano

file_name <- "Stroke_data.csv"
stroke_data <- read_csv(file_name)

set.seed(10, sample.kind="Rounding")

#######################################################################################
#                               Stroke prediction                                     #
#######################################################################################

str(stroke_data)

# All stroke cases found in the data set and their percentage in regard of the observations: 

sum(stroke_data$stroke)
mean(stroke_data$stroke)


#Summary of the data:
summary(stroke_data)



data <- stroke_data %>% 
  mutate(bmi = as.numeric(bmi), hypertension = as.factor(hypertension), heart_disease = as.factor(heart_disease), stroke = as.factor(ifelse(stroke == 1, "YES", "NO"))) %>% 
  na.omit(bmi) %>%
  mutate_if(is.character, as.factor)%>%
  dplyr::select(!id)

is.null(data)
summary(data)

#######################
#  Plots
data_stroke <- data %>% filter(stroke == "YES")

par(mfrow=c(2,2))
p1 <- qplot(data$age, data$bmi, xlab = "Age", ylab ="BMI")
p2 <- qplot(data_stroke$age, data_stroke$bmi, xlab = "Age with stroke", ylab ="BMI with stroke")
p3 <- qplot(data$age, data$avg_glucose_level, xlab = "Age", ylab ="Glucose level")
p4 <- qplot(data_stroke$age, data_stroke$avg_glucose_level, xlab = "Age with stroke", ylab ="Glucose level with stroke")

data %>% group_by(stroke) %>% summarise(avg_bmi = mean(bmi))

grid.arrange(p1, p2, ncol=2)
grid.arrange(p3, p4, ncol=2)

# Correlation

df <- data %>% 
  mutate(stroke = as.numeric(stroke))%>% 
  select(stroke,age,avg_glucose_level,bmi)
cor_matrix <- cor(df$stroke, df)
cor_matrix


## Plots of categorical variables

# ever_married
data %>% filter(stroke == "YES" ) %>%
  group_by(ever_married) %>% 
  summarise(n = n()) %>% 
  ggplot(aes(ever_married, n, fill = ever_married))+
  geom_col() + 
  geom_text(aes(label = n), vjust = -0.5, colour = "black")


# smoking_status
stroke_data %>% 
  filter(stroke == 1) %>%
  group_by(smoking_status) %>%
  summarise(n = n()) %>%
  ggplot(aes(smoking_status, n)) + 
  geom_col() + 
  geom_text(aes(label = n), vjust = 3, colour = "white")


# Urban or rural 

data %>% filter(stroke == "YES" ) %>% 
  group_by(Residence_type) %>% 
  summarise(n = n()) %>% 
  ggplot(aes(x = Residence_type, y = n, fill = Residence_type)) +
  geom_col() +
  geom_text(aes(label = n), vjust = 3, colour = "black")


# Stroke in gender

data %>% filter(stroke == "YES" ) %>% 
  group_by(gender) %>% 
  summarise(n = n()) %>% 
  ggplot(aes(x = gender, y = n, fill = gender)) +
  geom_col() +
  geom_text(aes(label = n), vjust = 3, colour = "black")

# Work type

data %>% filter(stroke == "YES") %>% 
  group_by(work_type) %>% 
  summarise(n = n()) %>%
  ggplot(aes(x = work_type, y = n, fill = work_type)) +
  xlab("Work Type") +
  geom_col() + 
  geom_text(aes(label = n), vjust = -0.5, colour = "black")

## Partitioning Data and building a model

test_index <- createDataPartition(y = data$stroke, times = 1, p = 0.2, list = FALSE)
train_set <- data[-test_index,]
test_set <- data[test_index,]


# Using General Logistic Regression model 

options(warn=-1)
train_glm <- train(stroke ~ ., data = train_set, method = 'glm', family= 'binomial')
pred_glm <- predict(train_glm, test_set)
cm_glm <- confusionMatrix(pred_glm, test_set$stroke)
cm_glm


# Random Forest: 

train_rf <- randomForest(stroke ~., data = train_set)
pred_rf <- predict(train_rf, test_set)
cm_rf <- confusionMatrix(pred_rf, test_set$stroke)
cm_rf


# Introducing ROSE library containing ovun sampling method. 
# "OVUN" stands for over-sampling minority examples (stroke positive) and under-sampling majority examples (stroke negative). 

ovun_set <- ovun.sample(stroke~.,
                    data = train_set,
                    method = "both",
                    N = 1000,
                    p = 0.5,
                    seed = 10)$data

# Decision Tree
fit <- rpart(stroke ~ ., data = ovun_set)
plot(fit, margin = 0.1)
text(fit,  cex = 0.6, minlength = 4)



# 10 folds validation with 10 repetition

control <- trainControl(method = "repeatedcv",
                   number = 10,
                   repeats = 10,
                   seed=10)


# Tuning the mtry for best value:

x <- seq(1,11,2)
acc_mtry <- lapply(x, function(xs){
  train(stroke ~ ., method = "rf", 
        data = {ovun.sample(stroke~.,
                            data = train_set,
                            method = "both",
                            N = 1000,
                            p = 0.5,
                            seed = 10)$data},
        tuneGrid = data.frame(mtry = xs))$results$Accuracy
})
plot(x,acc_mtry)




# with ovun dataset GLM:

train_glm <- train(stroke ~ ., data = ovun_set, method = 'glm', family= 'binomial')
pred_glm <- predict(train_glm, test_set)
cm_glm <- confusionMatrix(pred_glm, test_set$stroke)
cm_glm


# Random forest model with ovun data set

model_rf <- randomForest(stroke~.,
                       data = ovun_set,
                       mtry=x[which.max(acc_mtry)],
                       trControl = control,
                       ntrees = 500,
                       seed = 10)

pred_rf <- predict(model_rf, test_set)
cm_rf <- confusionMatrix(pred_rf, test_set$stroke)
cm_rf
