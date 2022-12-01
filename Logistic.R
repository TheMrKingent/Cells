library(readr)
library(caret)
library(glmnet)
library(reshape) 
library(ROCit)

setwd("C:/ARCHIVIO/2 - Magistrale/Semestre 3/Advanced statistical modelling fot Big Data/Progetto/Cells")

dat <- read_csv("C:/ARCHIVIO/2 - Magistrale/Semestre 3/Advanced statistical modelling fot Big Data/Progetto/Cells/dataset/dat_inverseStain.csv")

set.seed(200)
noise <- matrix(sample(0:10, 27558*1024, replace = T), nrow = 27558)
dat[,-1] <- dat[,-1] + noise


# Split train test
idx <- createDataPartition(dat$label, times=2, p=0.7)
train <- dat[idx$Resample1, ]
test <- dat[-idx$Resample1, ]

# Plot test
mat <- matrix(as.numeric(dat[20001,-2]), ncol=32, byrow = T)
data_melt <- melt(mat)  
ggplot(data_melt, aes(X1, X2)) +
  geom_tile(aes(fill = value)) +
  theme_void()



# ---------------------
# Regressione Logistica
fit1 <- glm(label~., family=binomial(link='logit'), data=train)
pred <- as.numeric(predict(fit1, test, type = 'response') > 0.5)

acc <- pred == test$label
mean(acc)  # Top: 0.899, inverseStain

# Accuracy solo 0-1 bit (Image mode '1'): 0.631
# Accuracy solo gray scale: 0.67
# Accuracy grayscale + contrasto: 0.66
# Accuracy grayscale + contrasto + mask: 0.895
# Accuracy grayscale + contrasto + mask + inverse: 0.899


# -------------------------------
# Regressione Logistica penalizz.

x <- model.matrix(label~., train)[,-1]
y <- train$label

cv.lasso <- cv.glmnet(x=x, y=y, alpha = 1, family = "binomial")
save(cv.lasso, file = 'CVobjects/cv_lasso_inverseStain.RData')
load(file = 'CVobjects/cv_lasso_inverseStain.RData')
plot(cv.lasso)

# Logistic no pen.
fit1 <- glmnet(x, y, alpha = 1, family = "binomial", lambda = 0)
pred <- as.numeric(predict(fit1, model.matrix(label~., test)[,-1], type='class'))
acc <- pred == test$label
mean(acc)


# Lambda min
fit_min <- glmnet(x, y, alpha = 1, family = "binomial", lambda = cv.lasso$lambda.min)
pred_min <- as.numeric(predict(fit_min, model.matrix(label~., test)[,-1], type='class'))
acc_min <- (pred_min == test$label)
mean(acc_min)  # Top: 0.901, bwstain

# Lambda 1se
fit_1se <- glmnet(x, y, alpha = 1, family = "binomial", lambda = cv.lasso$lambda.1se)
pred_1se <- as.numeric(predict(fit_1se, model.matrix(label~., test)[,-1], type='class'))
acc_1se <- (pred_1se == test$label)
mean(acc_1se)  # Top: 0.896, bwstain


#------------
# Plot exp(coeff. -1)*100

mat <- matrix((exp(coefficients(fit1)[-1])-1)*100, ncol=32, byrow = T)
data_melt <- melt(abs(mat))  

ggplot(data_melt, aes(X1, X2)) +
  geom_tile(aes(fill = value)) +
  theme_void()

mat <- matrix((exp(coefficients(fit_min)[-1])-1)*100, ncol=32, byrow = T)
data_melt <- melt(abs(mat))  

ggplot(data_melt, aes(X1, X2)) +
  geom_tile(aes(fill = value)) +
  theme_void()

mat <- matrix((exp(coefficients(fit_1se)[-1])-1)*100, ncol=32, byrow = T)
data_melt <- melt(abs(mat))  

ggplot(data_melt, aes(X1, X2)) +
  geom_tile(aes(fill = value)) +
  theme_void()

#----------
# ROC curve optimal cutoff

confusionMatrix(as.factor(pred_1se), as.factor(test$label), positive='1')  # 0.899
fitted_1se_prob <- as.numeric(predict(fit_1se, model.matrix(label~., test)[,-1], type='response'))

roc <- rocit(score=fitted_1se_prob, class=test$label)
p <- plot(roc, YIndex=TRUE)
p
p$AUC  # Area under the curve
cutoff <- p$`optimal Youden Index point`['cutoff']

pred_1se_optimal <- ifelse(fitted_1se_prob >= cutoff, 1, 0)
confusionMatrix(as.factor(pred_1se_optimal), as.factor(test$label), positive='1')  # 0.924














