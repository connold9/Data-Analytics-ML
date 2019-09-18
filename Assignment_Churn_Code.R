### Section 1 - Workspace and Packages ###
setwd("~/Documents/Fourth Year/Data Analytics/Assignment")

library(readxl)
library(janitor)
library(VIM)
library(ggplot2)
library(dplyr)
library(caret)
library(stringr)
library(gbm)
library(tibble)
library(ROCR)
library(xgboost)

##Removing Scientific notation for graphing purposes -  to revert set scipen = 0
options(scipen=999)

### Section 2 - Reading and Formatting Data ###
churn <- read_excel("churn .xlsx")
churn <- as.data.frame(unclass(churn))

churn$SeniorCitizen <- as.factor(churn$SeniorCitizen)

##Double checking if there are any duplicates - there isnt. 
chdup <- duplicated(churn)
table(chdup)

##Can remove both ID fields from the datasets
churn <- churn[,-1]

##Check for missing data using VIM
churn_aggr <- aggr(churn)

##As so few cases have missing values (11), just remove those cases.
table(complete.cases(churn))
churn <- churn[complete.cases(churn), ]

##Section 3 - PRELIMINARY ANALYSIS FOR CHURN DATA ###
##Variable 1 for Churn - Bar Charts to show groups more likely to churn/not churn
ggplot(churn) +
  geom_bar(aes_string(x="Dependents", fill="Churn"), position = "dodge")

ggplot(churn) +
  geom_bar(aes_string(x="Partner", fill="Churn"), position = "dodge")

ggplot(churn) +
  geom_bar(aes_string(x="PaymentMethod", fill="Churn"), position = "dodge")

ggplot(churn) +
  geom_bar(aes_string(x="InternetService", fill="Churn"), position = "dodge")

ggplot(churn) +
  geom_bar(aes_string(x="gender", fill="Churn"), position = "dodge")

##Senior Citizen analysis
churn %>%
  group_by(SeniorCitizen) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))

churn %>%
  group_by(SeniorCitizen, Churn) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))

ggplot(churn) +
  geom_bar(aes(x = SeniorCitizen, fill = Churn), position = "dodge")

##Loess Attempt
##Tenure and Dependents Loess
ggplot(churn, aes(tenure, as.numeric(Churn)-1, color=Dependents)) +
  stat_smooth(method="glm", formula=y~x,
              alpha=0.2, size=2, aes(fill=Dependents)) +
  geom_point(position=position_jitter(height=0.03, width=0)) +
  xlab("Total Charges") + ylab("Pr (Churn)")

##TotalCharges and Dependents Loess
ggplot(churn, aes(TotalCharges, as.numeric(Churn)-1, color=Dependents)) +
  stat_smooth(method="glm", formula=y~x,
              alpha=0.2, size=2, aes(fill=Dependents)) +
  geom_point(position=position_jitter(height=0.03, width=0)) +
  xlab("Total Charges") + ylab("Pr (Churn)")

##Loess Plot of Monthly Charges against Probability of Churn by Internet Service type
ggplot(churn, aes(MonthlyCharges, as.numeric(Churn)-1, color=InternetService)) +
  stat_smooth(method="glm", formula=y~x,
              alpha=0.2, size=2, aes(fill=InternetService)) +
  geom_point(position=position_jitter(height=0.03, width=0)) +
  xlab("Monthly Charges") + ylab("Pr (Churn)")

##Monthly Charges against Probability of Churn by Payment Method and Internet Service
ggplot(churn, aes(MonthlyCharges, as.numeric(Churn)-1, color=PaymentMethod)) +
  stat_smooth(method="glm", formula=y~x,
              alpha=0.2, size=2, aes(fill=InternetService)) +
  geom_point(position=position_jitter(height=0.03, width=0)) +
  xlab("Monthly Charges") + ylab("Pr (Churn)")

##VARIABLE - Churn vs TotalCharges by Gender
ggplot(churn, aes(as.character(Churn), TotalCharges, color = as.character(gender))) + 
  geom_boxplot()

##VARIABLE - Churn vs Tenure by SeniorCitizen
ggplot(churn, aes(as.character(Churn), tenure, color = as.character(SeniorCitizen))) + 
  geom_boxplot()

ggplot(churn, aes(as.character(Churn), TotalCharges, color = as.character(InternetService))) + 
  geom_boxplot()

##Get people moved from Electronic CHeck to another form of payment!
ggplot(churn) +
  geom_bar(aes(x=PaymentMethod,fill=Churn), position = "dodge")

##Breakdown of form of payment
churn %>%
  group_by(PaymentMethod) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))

churn %>%
  group_by(PaymentMethod, Churn) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))

ggplot(churn) +
  geom_bar(aes(x=PaymentMethod,fill=SeniorCitizen), position = "dodge")

cdplot(Churn ~ MonthlyCharges, churn, col=c("cornflowerblue", "orange"), main="Density Plot - Churn against Monthly Charges")
cdplot(Churn ~ TotalCharges, churn, col=c("cornflowerblue", "orange"), main="Density Plot - Churn against Total Charges")
cdplot(Churn ~ tenure, churn, col=c("cornflowerblue", "orange"), main="Density Plot - Churn against Tenure")

### Section 4 - Model Building (GBM)
## Changing churn from Yes/No to 1/0
## Removing total charges
churn <- churn[,-19]
churn <- churn %>%
  mutate(Churn = ifelse(Churn == "No",0,1))

intrain = createDataPartition(churn$Churn, p=0.7, list = F)
ctrainSet = churn[intrain,]
ctestSet = churn[-intrain,]

model_weights <- ifelse(ctrainSet$Churn == 1,
                        (1/table(ctrainSet$Churn)[2]) * 0.5,
                        (1/table(ctrainSet$Churn)[1]) * 0.5)

set.seed(123)
boosted_3k=gbm(formula  = Churn ~ . ,data = ctrainSet,distribution = "bernoulli",n.trees = 100,
               shrinkage = 0.1, interaction.depth = 6, cv.folds = 2)

#display of which variables have the most influence on the model
summary(boosted_3k)

pred = predict(object = boosted_3k, newdata = ctestSet, n.trees = 100, type = "response")
p <- ifelse(pred < 0.5, 0,1)
result <- data.frame(actual = ctestSet$Churn, pred = p)
result_t <- table(result)

correct = result_t[1,1] + result_t[2,2]
incorrect = sum(result_t)
sensit <- result_t[2,2]/(result_t[2,2] + result_t[2,1])
specif <- result_t[1,1]/(result_t[1,1] + result_t[1,2])
correct/incorrect * 100
result$actual <- as.factor(result$actual)
result$pred <- as.factor(result$pred)

## https://www.kaggle.com/aljaz91/ibm-s-attrition-tackling-class-imbalance-with-gbm ##
ggplot(result, 
        aes(x = pred, group = actual)) + 
        geom_bar(aes(y = ..prop.., fill = factor(..x..)), 
                   stat="count", 
                   alpha = 0.7) +
        geom_text(aes(label = scales::percent(..prop..), y = ..prop.. ), 
                   stat= "count", 
                   vjust = -.5) +
        labs(y = "Percentage", fill= "Actual") +
        facet_grid(~actual) +
        scale_fill_manual(values = c("#386cb0","#fdb462")) + 
        theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) + 
        ggtitle("Actual Churn")

#https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
#Description of ROC % AUC
ROC = prediction(pred, ctestSet$Churn)
plot(performance(ROC, "tpr", 'fpr'))
gbm.perf(boosted_3k)
gbm.roc.area(ctestSet$Churn, pred)

#make a gridsearch to find best values
hyper_grid <- expand.grid(
  shrinkage = c(0.003, 0.01, 0.05),
  interaction.depth = c(2, 4),
  #n.trees = c(1000, 3000),
  #n.minobsinnode = c(5, 10, 15),
  #bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  AUC = 0                     # a place to dump results
  
)

for(i in 1:nrow(hyper_grid)) {
  gbm.tune <- gbm(
    formula = Churn ~ .,
    distribution = "bernoulli",
    data = ctrainSet,
    n.trees = 2956, #hyper_grid$n.trees[i],
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = 15, #hyper_grid$n.minobsinnode[i],
    bag.fraction = 0.65, #hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = T,
    weights = model_weights
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] = which.min(gbm.tune$valid.error)
  hyper_grid$AUC[i] = gbm.roc.area(ctestSet$Churn, predict(object = gbm.tune, newdata = ctestSet, n.trees = 2956))
}

hyper_grid[which.max(hyper_grid$AUC),]
## Original grid search indicated a lower number of trees and lower shrinkage is best
## Second grid search indicated even lower shrinkage, increased trees is better
## Settled on N.Trees 2716, Depth 2 and Shrinkage 0.003 - AUC 0.8513
## Min Obs 15, Bag Frac 0.65

boosted_3k_crossval = gbm(formula  = Churn ~ . ,data = ctrainSet, distribution = "bernoulli",n.trees = 2716,
                          shrinkage = 0.003, interaction.depth = 2, bag.fraction = 0.65, n.minobsinnode = 15, weights = model_weights)

summary(boosted_3k_crossval)
predcrossval = predict(object = boosted_3k_crossval, newdata = ctestSet, n.trees = 2716)
ROC_crossval = prediction(predcrossval, ctestSet$Churn)
plot(performance(ROC_crossval, "tpr", 'fpr'))
gbm.perf(boosted_3k_crossval)
gbm.roc.area(ctestSet$Churn, predcrossval)

p <- predict(boosted_3k_crossval, newdata = ctestSet, type = "response", n.trees = 2716)
p <- ifelse(p < 0.5, 0,1)
result <- data.frame(actual = ctestSet$Churn, pred = p)
result_t <- table(result)
correct = result_t[1,1] + result_t[2,2]
sensit <- result_t[2,2]/(result_t[2,2] + result_t[1,2])
specif <- result_t[1,1]/(result_t[1,1] + result_t[2,1])
churn_accuracy <- result_t[2,2]/(result_t[2,2] + result_t[2,1])
incorrect = sum(result_t)
correct/incorrect * 100

result$actual <- as.factor(result$actual)
result$pred <- as.factor(result$pred)

## https://www.kaggle.com/aljaz91/ibm-s-attrition-tackling-class-imbalance-with-gbm ##
ggplot(result, 
        aes(x = pred, group = actual)) + 
        geom_bar(aes(y = ..prop.., fill = factor(..x..)), 
                   stat="count", 
                   alpha = 0.7) +
        geom_text(aes(label = scales::percent(..prop..), y = ..prop.. ), 
                   stat= "count", 
                   vjust = -.5) +
        labs(y = "Percentage", fill= "Actual") +
        facet_grid(~actual) +
        scale_fill_manual(values = c("#386cb0","#fdb462")) + 
        theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) + 
        ggtitle("Actual Churn")

# Use ROCR package to plot ROC Curve
p <- predict(boosted_3k_crossval, newdata = ctestSet, type = "response", n.trees = 2716)
gbm.pred <- prediction(p, ctestSet$Churn)
gbm.perf <- performance(gbm.pred, "tpr", "fpr")

plot(gbm.perf,
     avg="threshold",
     #colorize=TRUE,
     col = "blue",
     lwd=1,
     main="ROC Curve w/ Thresholds",
     #print.cutoffs.at=seq(0, 1, by=0.05),
     text.adj=c(-0.5, 0.5),
     text.cex=0.5)
grid(col="lightgray")
axis(1, at=seq(0, 1, by=0.1))
axis(2, at=seq(0, 1, by=0.1))
abline(v=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
abline(h=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
lines(x=c(0, 1), y=c(0, 1), col="black", lty="dotted")

### Section 5 - Model Building (XGBoost)
##There is a lot of preparation to be done - will start with the Churn dataset
##Factor variables need to be recoded using hot-code variables (indicators in a sparse matrix)

library(Matrix)
library(magrittr)
library(xgboost)

##Moving Churn to first column
col_idx <- grep("Churn", names(churn))
churn <- churn[, c(col_idx, (1:ncol(churn))[-col_idx])]
names(churn)

#And making churn 0 for No and 1 for yes
churn <- churn %>%
  mutate(Churn = ifelse(Churn == "No",0,1))

##Using SMOTE
library("DMwR")
churn$Churn <- as.factor(churn$Churn)
newChurnData <- SMOTE(Churn ~ ., churn, perc.over = 200,perc.under=150)
table(newChurnData$Churn)
newChurnData$Churn <- as.numeric(newChurnData$Churn)
newChurnData$Churn <- as.numeric(newChurnData$Churn-1)
summary(newChurnData$Churn)
##New Churn data is an equally weighted (5607 each) version of the Churn data

##Not using SMOTE
newChurnData = churn

##XGB for CHURN data after SMOTE
test_rows = sample.int(nrow(newChurnData), nrow(newChurnData)/3)
churn_test = newChurnData[test_rows,]
churn_train = newChurnData[-test_rows,]

ctrainm <- sparse.model.matrix(Churn ~ . -1 , data = churn_train)
ctrain_label <- churn_train[,"Churn"]
ctrain_matrix <- xgb.DMatrix(data = as.matrix(ctrainm), label = ctrain_label)

ctestm <- sparse.model.matrix(Churn ~ .-1, data = churn_test)
ctest_label <- churn_test[,"Churn"]
ctest_matrix <- xgb.DMatrix(data = as.matrix(ctestm), label = ctest_label)

#Parameters 
nc <- length(unique(ctrain_label))
xgb_params <- list("objective" = "binary:logistic",
                   "eval_metric" = "auc")
##NOTE - AUC doesn't work with Multi:Softmax, so changed to binary logistic!
cwatchlist <- list(train = ctrain_matrix, test = ctest_matrix)

set.seed(123)
#Building the xgb model - defaults
bst_model <- xgb.train(params = xgb_params,
                       nrounds = 100,
                       data = ctrain_matrix, 
                       watchlist = cwatchlist,
                       #scale_pos_weight = 3,
                       early_stopping_rounds = 5)

importancedef <- xgb.importance(feature_names = ctrainm@Dimnames[[2]], model = bst_model)
importance_smote <- xgb.importance(feature_names = ctrainm@Dimnames[[2]], model = bst_model)

##Prediction and confusion matrix
p <- predict(bst_model, newdata = ctest_matrix)
pred <- ifelse(p<0.5,0,1)

results_df <- data.frame(Actual = ctest_label, Pred = pred)
result <- table(results_df)

correct = result[1,1] + result[2,2]
incorrect = sum(result)
accuracy <- correct/incorrect
sensitivity <- result[2,2]/(result[1,2] + result[2,2])
specificity <- result[1,1]/(result[2,1]+result[1,1])
results_df$Actual <- as.factor(results_df$Actual)
results_df$Pred <- as.factor(results_df$Pred)
table(results_df)

ggplot(results_df) +
  geom_bar(aes_string(x="Actual", fill= "Pred"),position = "dodge")

ggplot(results_df, 
        aes(x = Pred, group = Actual)) + 
        geom_bar(aes(y = ..prop.., fill = factor(..x..)), 
                   stat="count", 
                   alpha = 0.7) +
        geom_text(aes(label = scales::percent(..prop..), y = ..prop.. ), 
                   stat= "count", 
                   vjust = -.5) +
        labs(y = "Percentage", fill= "Predicted") +
        facet_grid(~Actual) +
        scale_fill_manual(values = c("#386cb0","#fdb462")) + 
        theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) + 
        ggtitle("Actual Churn")


hyper_grid <- expand.grid(
  subsample = c(.9), 
  colsample_bytree = c(1),
  gamma = c(0,100,500),
  minchild = c(1),
  optimal_trees = 0,            
  AUC = 0                    
)

for(i in 1:nrow(hyper_grid)) {
    
    set.seed(123)
    loop_model <- xgb.train(params = xgb_params, 
                           data = ctrain_matrix, 
                           nrounds = 500,
                           eta = 0.07, ##0.3 ,default - low = more robust to overfitting
                           watchlist = cwatchlist,
                           max.depth = 8,
                           min_child_weight = hyper_grid$minchild[i],
                           subsample = hyper_grid$subsample[i],
                           colsample_bytree = hyper_grid$colsample_bytree[i],
                           verbose = T,
                           early_stopping_rounds = 20) 
    
    ##Prediction and confusion matrix
    hyper_grid$AUC[i] <- max(loop_model$evaluation_log$test_auc)
    hyper_grid$optimal_trees[i] <- loop_model$best_iteration
  
}


hyper_grid %>%
  dplyr::arrange(desc(AUC)) %>%
  head(10)

## First Grid Serach - 363 trees, Eta 0.1, Depth of 6, AUC 0.940013
## Second Grid Search - 387 Trees, Eta 0.08, Depth of 7, AUC 0.940806
## Third Grid Search - 496 Trees, Eta 0.07, Depth of 8, AUC 0.94234
## Fourth Grid Search - Subsample 0.9, Optimal Trees 427, Colsample =1, Minchild = 1, AUC 0.94402
## Fifth Grid Search - Gamma has no effect range from 30-70 - AUC 0.94402

set.seed(123)
final_model <- xgb.train(params = xgb_params, 
                           data = ctrain_matrix, 
                           nrounds = 450,
                           eta = 0.07, ##0.3 ,default - low = more robust to overfitting
                           watchlist = cwatchlist,
                           max.depth = 8,
                           gamma = 0,
                           min_child_weight = 1,
                           subsample = 0.9,
                           colsample_bytree = 1,
                           verbose = T,
                           early_stopping_rounds = 20) 

max(final_model$evaluation_log$test_auc)
which.max(final_model$evaluation_log$test_auc)

importancefinal <- xgb.importance(feature_names = ctrainm@Dimnames[[2]], model = final_model)

##Prediction and confusion matrix
p <- predict(final_model, newdata = ctest_matrix)
pred <- ifelse(p<0.5,0,1)
results_df <- data.frame(Actual = ctest_label, Pred = pred)
result <- table(results_df)

correct = result[1,1] + result[2,2]
incorrect = sum(result)
accuracy <- correct/incorrect
sensitivity <- result[2,2]/(result[1,2] + result[2,2])
specificity <- result[1,1]/(result[2,1]+result[1,1])

results_df$Actual <- as.factor(results_df$Actual)
results_df$Pred <- as.factor(results_df$Pred)
table(results_df)

ggplot(results_df, 
        aes(x = Pred, group = Actual)) + 
        geom_bar(aes(y = ..prop.., fill = factor(..x..)), 
                   stat="count", 
                   alpha = 0.7) +
        geom_text(aes(label = scales::percent(..prop..), y = ..prop.. ), 
                   stat= "count", 
                   vjust = -.5) +
        labs(y = "Percentage", fill= "Actual") +
        facet_grid(~Actual) +
        scale_fill_manual(values = c("#386cb0","#fdb462")) + 
        theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) + 
        ggtitle("Actual Churn")

##Drawing ROC
library(ROCR)

# Use ROCR package to plot ROC Curve
xgb.pred <- prediction(p, ctest_label)
xgb.perf <- performance(xgb.pred, "tpr", "fpr")

plot(xgb.perf,
     add= T,
     avg="threshold",
     #colorize=TRUE,
     col = "red",
     lwd=1,
     main="ROC Curve w/ Thresholds",
     #print.cutoffs.at=seq(0, 1, by=0.05),
     text.adj=c(-0.5, 0.5),
     text.cex=0.5)
grid(col="lightgray")
axis(1, at=seq(0, 1, by=0.1))
axis(2, at=seq(0, 1, by=0.1))
abline(v=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
abline(h=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
lines(x=c(0, 1), y=c(0, 1), col="black", lty="dotted")
