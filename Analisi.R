library(readr)
library(caret)
library(glmnet)
setwd("C:/ARCHIVIO/2 - Magistrale/Semestre 3/Advanced statistical modelling fot Big Data/Progetto/Cells")

dat <- read_csv("C:/ARCHIVIO/2 - Magistrale/Semestre 3/Advanced statistical modelling fot Big Data/Progetto/Cells/dat.csv")

# Split train test
set.seed(100)
idx <- createDataPartition(dat$label, times=2, p=0.7)
train <- dat[idx$Resample1, ]
test <- dat[-idx$Resample1, ]


# ---------------------
# Regressione Logistica
fit1 <- glm(label~., family=binomial(link='logit'), data=train)
pred <- as.numeric(predict(fit1, test, type = 'response') > 0.5)

acc <- pred == test$label
mean(acc)  # ~0.65


# -------------------------------
# Regressione Logistica penalizz.

x <- model.matrix(label~., train)[,-1]
y <- train$label

cv.lasso <- cv.glmnet(x=x, y=y, alpha = 1, family = "binomial")
save(cv.lasso, file = 'cv_lasso.RData')
#load(file = 'cv_lasso.RData')
plot(cv.lasso)

# Lambda min
fit_min <- glmnet(x, y, alpha = 1, family = "binomial", lambda = cv.lasso$lambda.min)
pred_min <- as.numeric(predict(fit_min, model.matrix(label~., test)[,-1], type='class'))
acc_min <- (pred_min == test$label)
mean(acc_min)  # ~0.66

# Lambda 1se
fit_1se <- glmnet(x, y, alpha = 1, family = "binomial", lambda = cv.lasso$lambda.1se)
pred_1se <- as.numeric(predict(fit_1se, model.matrix(label~., test)[,-1], type='class'))
acc_1se <- (pred_1se == test$label)
mean(acc_1se)  # ~0.66


