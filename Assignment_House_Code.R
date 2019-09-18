### Section 1 - Workspace and Packages ###
dir = "C:/Users/Devin/Documents/Fourth Year Assorted/Data Analytics/Assignment"
setwd(dir)

library(readxl)
library(janitor)
library(VIM)
library(ggplot2)
library(dplyr)
library(readxl)
library(caret)
library(stringr)
library(gbm)
library(tibble)
library(ROCR)
library(xgboost)
library(ModelMetrics)
library(MLmetrics)
library(car)

##Removing Scientific notation for graphing purposes -  to revert set scipen = 0
options(scipen=999)

### Section 2 - Reading and Formatting Data ###
house <- read_excel("Kingscounty house data.xlsx")
house <- as.data.frame(unclass(house))

house$condition <- as.factor(house$condition)
house$zipcode <- as.factor(house$zipcode)
house$yr_renovated <- as.factor(house$yr_renovated)
house$yr_built <- as.factor(house$yr_built)
house$grade <- as.factor(house$grade)
house$view <- as.factor(house$view)
house$waterfront <- as.factor(house$waterfront)
house$floors <- as.factor(house$floors)

#Recode Date
house$year = substr(house$date, 1, 4)
house$month = substr(house$date, 5, 6)
house$day = substr(house$date, 7, 8)

#Drop Date 
drops = c('date')
house = house[ , !(names(house) %in% drops)]

#Check variable types
sapply(house, class)

#Recode Classes
house$year = as.integer(house$year)
house$month = as.integer(house$month)
house$day = as.integer(house$day)

#Derived Variables - Using Year Built and Year Renovated to create Age
house$latest_age = ifelse(house$yr_renovated == 0, house$yr_built, house$yr_renovated)
house$age = house$year-house$latest_age

##Can remove both ID fields from the datasets
house <- house[,-1]

##Double checking if there are any duplicates - there isnt. 
hodup <- duplicated(house)
table(hodup)

##Check for missing data using VIM - no missing cases in the housing dataset
house_aggr <- aggr(house)

summary(house$price)
length(levels(house$zipcode))
### Section 3 - Descriptive and Summary Analysis ###

##Pick three variables to analyse compared to price and compared to churn.
##Important to use both numerical and categorical variables
### Variable 1 - Square Foot Living vs Price ###
gg <- ggplot(house, aes(x=sqft_living, y=price)) + 
  geom_point(aes(col=grade)) + 
  geom_smooth(method="loess", se=F) + 
  labs(subtitle="Price Vs Square Feet", 
       y="Price", 
       x="Square Foot of Living Space", 
       title="Scatterplot")
gg

##Grouping Grades to make for more easily visible graph
house$'new_grade' <- house$grade

##Recoding the factors 1-6 as <7 to make the plot more readable
house$new_grade <- recode(house$grade, "c('1','2','3','4','5','6')='<7'")
house$new_grade <- relevel(house$new_grade, "13")
house$new_grade <- relevel(house$new_grade, "12")
house$new_grade <- relevel(house$new_grade, "11")
house$new_grade <- relevel(house$new_grade, "10")
house$new_grade <- relevel(house$new_grade, "9")
house$new_grade <- relevel(house$new_grade, "8")
house$new_grade <- relevel(house$new_grade, "7")
house$new_grade <- relevel(house$new_grade, "<7")

##A function to dispaly the number of observations in my boxplot
give.n <- function(x){
  return(c(y = median(x)*0, label = length(x) ) )
  # experiment with the multiplier to find the perfect position
}

##VARIABLE 2 - Categorical - Grade and Condition vs Price
##Boxplot of Grade, split by condition, vs Price, with # of observations shown in red
g <- ggplot(house, aes(new_grade, price)) +
  geom_boxplot(aes(fill=factor(condition))) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  stat_summary(fun.data = give.n, geom = "text", fun.y = median, 
               position = position_dodge(width = 0.01),
               hjust = 0.5,
               vjust = 0.9,
               col = "blue") +
  labs(title="Box plot", 
       subtitle="Price by Grade of House, Split by Condition",
       x="Grade of Property",
       y="Price",
       fill = "Condition")
g


##VARIABLE 3 - Categorical - ZIPCODE vs PRICE
# first we use dplyr to calculate the mean price and SE for each zipcode
zipdata <- house %>% group_by(zipcode) %>%
  summarise(M = mean(price, na.rm=T),
            SE = sd(price, na.rm=T)/sqrt(length(na.omit(price))),
            N = length(na.omit(price)))

# make zipcode into an ordered factor, ordering by mean price:
zipdata$zipcode <- factor(zipdata$zipcode)
zipdata$zipcode <- reorder(zipdata$zipcode, zipdata$M)

##Taking a sample of 15 from this zipcode data for graphing purposes
##can see a strong link between mean house prices and zipcode
zipdata <- sample_n(zipdata, 15)

ggplot(zipdata, aes(x = M, xmin = M-SE, xmax = M+SE, y = zipcode )) +
  geom_point() + geom_segment( aes(x = M-SE, xend = M+SE,
                                   y = zipcode, yend=zipcode)) +
  theme_bw() + xlab("Mean Price") + ylab("Zipcode") +
  ggtitle("House Price by Zipcode, with SE")

### Section 4 - Model Building (GBM) ###
intrain = createDataPartition(house$price, p=0.7, list = F)
htrainSet = house[intrain,]
htestSet = house[-intrain,]

house_train = as.data.frame(htrainSet)
house_test = as.data.frame(htestSet)

## Default Model ##
initial_gbm_house = gbm(price ~ . ,
          data = house_train,
          distribution = "gaussian",
          n.trees = 100,
          interaction.depth = 1,
          shrinkage = 0.1,
          n.cores = NULL, # will use all cores by default
          verbose = FALSE)

print(initial_gbm_house)

### Section 4.1 Model Graph ###

p_house_gbm = predict(initial_gbm_house, initial_gbm_house$n.trees, newdata = house_test)

results_df_house_gbm = data.frame(Actual = house_test$price, Pred = p_house_gbm)
result_gbm = table(results_df_house_gbm)

gg <- ggplot(results_df_house_gbm, aes(x=Actual, y=Pred)) + 
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(subtitle="Actual vs Predicted", 
       y="Predicted", 
       x="Actual", 
       title="Scatterplot")
gg

### Section 4.2 - Model Fit Error ###
model_mape_house_gbm = MAPE(results_df_house_gbm$Pred, results_df_house_gbm$Actual)

percent_accuracy_house_gbm = 1-model_mape_house_gbm

model_rmse_house_gbm = rmse(results_df_house_gbm$Actual, results_df_house_gbm$Pred)

r_sq_house_gbm = cor(results_df_house_gbm$Actual, results_df_house_gbm$Pred)^2

print("Initial model results")
print(paste0("Percent Accuracy Is: " , percent_accuracy_house_gbm))
print(paste0("RMSE is: ", model_rmse_house_gbm))
print(paste0("R Squared is: ", r_sq_house_gbm))

par(mar = c(5, 8, 1, 1))
summary(
  initial_gbm_house, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

### Section 4.3 - Log Transform Model ###

# Log transforming price

gbm_house_v1.2 = gbm(log(price) ~ . ,
                        data = house_train,
                        distribution = "gaussian",
                        n.trees = 100,
                        interaction.depth = 1,
                        shrinkage = 0.1,
                        n.cores = NULL, # will use all cores by default
                        verbose = FALSE)

summary(
  gbm_house_v1.2, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

p_house_gbm = predict(gbm_house_v1.2, gbm_house_v1.2$n.trees, newdata = house_test)

results_df_house_gbm_v1.2 = data.frame(Actual = house_test$price, Pred = exp(p_house_gbm))
result_gbm_v1.2 = table(results_df_house_gbm_v1.2)
results_df_house_gbm_v1.2 <- exp(results_df_house_gbm_v1.2)
plot(results_df_house_gbm_v1.2$Actual, results_df_house_gbm_v1.2$Pred, xlab = "Actual Price", ylab = "Predicted Price", main = "House Gradient Boosting - Hyper Tune")

gg <- ggplot(results_df_house_gbm_v1.2, aes(x=Actual, y=Pred)) + 
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(subtitle="Actual vs Predicted", 
       y="Predicted", 
       x="Actual", 
       title="Scatterplot")
gg

model_mape_house_gbm_v1.2 = MAPE(results_df_house_gbm_v1.2$Pred, results_df_house_gbm_v1.2$Actual)
percent_accuracy_house_gbm_v1.2 = 1-model_mape_house_gbm_v1.2

model_rmse_house_gbm_v1.2 = rmse(results_df_house_gbm_v1.2$Actual, results_df_house_gbm_v1.2$Pred)
r_sq_house_gbm_v1.2 = cor(results_df_house_gbm_v1.2$Actual, results_df_house_gbm_v1.2$Pred)^2

print("Initial model results")
print(paste0("Percent Accuracy Is: " , percent_accuracy_house_gbm_v1.2))
print(paste0("MAPE is: " , model_mape_house_gbm_v1.2))
print(paste0("RMSE is: ", model_rmse_house_gbm_v1.2))
print(paste0("R Squared is: ", r_sq_house_gbm_v1.2))

# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm_house_v1.2)

##Using Log Price is better!

### Section 5 - Parameter Tuning for GBM
#Using a grid to tune
hyper_grid <- expand.grid(
  shrinkage = c( .02),
  interaction.depth = c(8),
  n.trees = c(1000),
  n.minobsinnode = c(1,10),
  bag.fraction = c(.9, 1), 
  optimal_trees = 0,             
  min_RMSE = 0
)

# grid search 
for(i in 1:nrow(hyper_grid)) {
  set.seed(123)
  # train model
  gbm.tune <- gbm(
    formula = log(price) ~ .,
    distribution = "gaussian",
    data = house_train,
    n.trees = hyper_grid$n.trees[i],
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  p_house_tune = predict(gbm.tune, gbm.tune$n.trees, newdata = house_test)
  RMSE <- rmse(exp(p_house_tune), house_test$price)
  hyper_grid$min_RMSE[i] <- RMSE
  print(min(gbm.tune$valid.error))
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  #hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)

## First result - Shrinkage 0.01, Depth 8, Trees 1500, RMSE 129,954
## Second Result - SHrinkage 0.02, Depth 8, Trees 939, RMSE 129,338
## Third Result - Min Obs 10 Bag Frac 0.9  - RMSE 126,058 (852 Trees)

set.seed(123)
gbm_house_vFINAL = gbm(log(price) ~ . ,
                      data = house_train,
                      distribution = "gaussian",
                      n.trees = 852,
                      interaction.depth = 8,
                      shrinkage = 0.02,
                      bag.fraction = .9,
                      n.minobsinnode = 10,
                      train.fraction = .75,
                      n.cores = NULL, # will use all cores by default
                      verbose = T)

p_house_gbm_final = predict(gbm_house_vFINAL, gbm_house_vFINAL$n.trees, newdata = house_test)

results_df_house_gbm_final = data.frame(Actual = house_test$price, Pred = exp(p_house_gbm_final))
result_gbm_final = table(results_df_house_gbm_final)
plot(results_df_house_gbm_final$Actual, results_df_house_gbm_final$Pred, xlab = "Actual Price", ylab = "Predicted Price", main = "House Gradient Boosting - Hyper Tune")

gg <- ggplot(results_df_house_gbm_final, aes(x=Actual, y=Pred)) + 
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(subtitle="Actual vs Predicted", 
       y="Predicted", 
       x="Actual", 
       title="Scatterplot")
gg

model_mape_house_gbm_final = MAPE(results_df_house_gbm_final$Pred, results_df_house_gbm_final$Actual)
percent_accuracy_house_gbm_final = 1-model_mape_house_gbm_final

model_rmse_house_gbm_final = rmse(results_df_house_gbm_final$Actual, results_df_house_gbm_final$Pred)
r_sq_house_gbm_final = cor(results_df_house_gbm_final$Actual, results_df_house_gbm_final$Pred)^2

print("Initial model results")
print(paste0("Percent Accuracy Is: " , percent_accuracy_house_gbm_final))
print(paste0("MAPE is: " , model_mape_house_gbm_final))
print(paste0("RMSE is: ", model_rmse_house_gbm_final))
print(paste0("R Squared is: ", r_sq_house_gbm_final))

# Variable Importance
par(mar = c(5, 8, 1, 1))
summary(
  gbm_house_vFINAL, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

### Section 6 - XGB For House ###
## Section 6.1 Dataset Set Up ##
set.seed(123) 
house$price <- log(house$price)
house$price <- exp(house$price)

sample = sample.int(n = nrow(house), size = floor(.8*nrow(house)), replace = F)
house_train = house[sample, ] #just the samples
house_test  = house[-sample, ] #everything but the samples

htrainm <- sparse.model.matrix(price ~ . -1 , data = house_train)
htrain_label <- house_train[,"price"]
htrain_matrix <- xgb.DMatrix(data = as.matrix(htrainm), label = htrain_label)

htestm <- sparse.model.matrix(price ~ .-1, data = house_test)
htest_label <- house_test[,"price"]
htest_matrix <- xgb.DMatrix(data = as.matrix(htestm), label = htest_label)

## Section 6.2 Initial Model ##
watchlist = list(train=htrain_matrix, test=htest_matrix)

set.seed(123)
bst = xgb.train(data = htrain_matrix,
                max.depth = 6, 
                eta = 0.3, 
                nround = 100, 
                watchlist = watchlist, 
                objective = "reg:linear", 
                early_stopping_rounds = 10)

importancedef <- xgb.importance(feature_names = htrainm@Dimnames[[2]], model = bst)

## Section 6.3 Initial Model Evaluation and Visualisation ##
p_house_xgb = predict(bst, bst$n.trees, newdata = htest_matrix)

results_df_house_xgb = data.frame(Actual = exp(house_test$price), Pred = exp(p_house_xgb))
#results_df_house_xgb = data.frame(Actual = house_test$price, Pred = p_house_xgb)

result_xgb = table(results_df_house_xgb)
plot(results_df_house_xgb$Actual, results_df_house_xgb$Pred, xlab = "Actual Price", ylab = "Predicted Price", main = "House XGB - Hyper Tune")

gg <- ggplot(results_df_house_xgb, aes(x=Actual, y=Pred)) + 
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(subtitle="Actual vs Predicted", 
       y="Predicted", 
       x="Actual", 
       title="Scatterplot")
gg


model_mape_house_xgb = MAPE(results_df_house_xgb$Pred, results_df_house_xgb$Actual)
percent_accuracy_house_xgb = 1-model_mape_house_xgb

model_rmse_house_xgb = rmse(results_df_house_xgb$Actual, results_df_house_xgb$Pred)
r_sq_house_xgb = cor(results_df_house_xgb$Actual, results_df_house_xgb$Pred)^2

print("Initial model results")
print(paste0("Percent Accuracy Is: " , percent_accuracy_house_xgb))
print(paste0("RMSE is: ", model_rmse_house_xgb))
print(paste0("R Squared is: ", r_sq_house_xgb))

## Section 6.4 Parameter Tuning ##
##Grid search hypertuning for xgb
## First run will use broader values - get an approximate then tune once again!
hyper_grid <- expand.grid(
  gamma = c(1,5),
  subsample = c(1), 
  colsample_bytree = c(1),
  minchild = c(1),
  optimal_trees = 0,              
  min_RMSE = 0,
  calc_RMSE = 0
)

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # create parameter list
  params <- list(
    eta = 0.03,
    max_depth = 7,
    min_child_weight = hyper_grid$minchild[i],
    gamma = hyper_grid$gamma[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  # reproducibility
  set.seed(123)
  
  # train model
  xgb.tune <- xgb.train(
    watchlist = watchlist,
    params = params,
    nrounds = 1000,
    data = htrain_matrix,
    objective = "reg:linear",  
    #verbose = 0,
    print_every_n = 500,
    early_stopping_rounds = 10
  )
  
  p_tune = predict(xgb.tune, xgb.tune$n.trees, newdata = htest_matrix)
  results_tune = data.frame(Actual = exp(house_test$price), Pred = exp(p_tune))
  tune_rmse = rmse(results_tune$Actual, results_tune$Pred)
  
  # add min training error and trees to grid
  
  hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_rmse)
  hyper_grid$min_RMSE[i] <- min(xgb.tune$evaluation_log$test_rmse)
  hyper_grid$calc_RMSE[i] <- tune_rmse
  print(paste0(tune_rmse, " ", i))
}

hyper_grid %>%
  dplyr::arrange(min_RMSE) %>%
  head(10)

#Eta 0.03 Depth 7 Trees 819 RMSE - 107382.8
#Default best for Min_Child, Subsample and Colsample

## Section 6.5 Final Model and Evaluation ##
set.seed(123)
bst_final = xgb.train(data = htrain_matrix,
                max.depth = 7, 
                eta = 0.03,
                nround = 819, 
                watchlist = watchlist, 
                objective = "reg:linear", 
                print_every_n = 5,
                subsample = 0.75,
                early_stopping_rounds = 50)

p_house_xgb_final = predict(bst_final, bst_final$n.trees, newdata = htest_matrix)

results_df_house_xgb_final = data.frame(Actual = exp(house_test$price), Pred = exp(p_house_xgb_final))

result_xgb_final = table(results_df_house_xgb_final)
plot(results_df_house_xgb_final$Actual, results_df_house_xgb_final$Pred, xlab = "Actual Price", ylab = "Predicted Price", main = "House XGB - Hyper Tune")

gg <- ggplot(results_df_house_xgb_final, aes(x=Actual, y=Pred)) + 
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(subtitle="Actual vs Predicted", 
       y="Predicted", 
       x="Actual", 
       title="Scatterplot")
gg

model_mape_house_xgb_final = MAPE(results_df_house_xgb_final$Pred, results_df_house_xgb_final$Actual)
percent_accuracy_house_xgb_final = 1-model_mape_house_xgb_final

model_rmse_house_xgb_final = rmse(results_df_house_xgb_final$Actual, results_df_house_xgb_final$Pred)
r_sq_house_xgb_final = cor(results_df_house_xgb_final$Actual, results_df_house_xgb_final$Pred)^2

print("Initial model results")
print(paste0("Percent Accuracy Is: " , percent_accuracy_house_xgb_final))
print(paste0("RMSE is: ", model_rmse_house_xgb_final))
print(paste0("R Squared is: ", r_sq_house_xgb_final))

importancedef <- xgb.importance(feature_names = htrainm@Dimnames[[2]], model = bst_final)