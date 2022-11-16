library(readr)
library(caret)
library(glmnet)
library(reshape)   
setwd("C:/ARCHIVIO/2 - Magistrale/Semestre 3/Advanced statistical modelling fot Big Data/Progetto/Cells")

dat <- read_csv("C:/ARCHIVIO/2 - Magistrale/Semestre 3/Advanced statistical modelling fot Big Data/Progetto/Cells/dataset/dat_inverseStain.csv")

noise <- matrix(sample(0:10, 27558*1024, replace = T), nrow = 27558)
dat[,-1] <- dat[,-1] + noise



# Split train test
set.seed(200)
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

# Accuracy solo gary scale: 0.67
# Accuracy senza inverse: --


# -------------------------------
# Regressione Logistica penalizz.

x <- model.matrix(label~., train)[,-1]
y <- train$label

cv.lasso <- cv.glmnet(x=x, y=y, alpha = 1, family = "binomial")
save(cv.lasso, file = 'cv_lasso_inverseStain.RData')
#load(file = 'cv_lasso_inverseStain.RData')
plot(cv.lasso)

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


#-----
# Plot coeff.

mat <- matrix(coefficients(fit_min)[-1], ncol=32, byrow = T)
data_melt <- melt(abs(mat))  

ggplot(data_melt, aes(X1, X2)) +
  geom_tile(aes(fill = value)) +
  theme_void()
