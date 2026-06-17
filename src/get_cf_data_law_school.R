# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

# We only need level 3 (structural) counterfactuals

# Seed
set.seed(42)

# Packages
library(dplyr)
library(caret)
# library(rstan)
# rstan_options(auto_write = TRUE)
# options(mc.cores = parallel::detectCores())
library(data.table)


# Setup -------------------------------------------------------------------

# working directory
dir = dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dir)
setwd('..')
wd = getwd()

# folder paths
path_data = paste(wd, "/data/", sep = "")
path_mdls = paste(wd, "/src/stan_models/", sep = "")
path_rslt = paste(wd, "/data/", sep = "")

# original data
org_df <- read.csv(file = paste(path_data, "LawSchool.csv", sep = ""), sep = '|')

# initial vars
use_race = "race_nonwhite"
vars <- c("LSAT", "UGPA", "sex")
vars <- append(vars, use_race)
# modified vars for counterfactual generation
vars_m <- c("LSAT", "UGPA")

# modeling data
df <- org_df %>% select(vars)

# var transformation
df$LSAT <- round(df$LSAT)

sense_cols <- c("female", "male")
print(sense_cols)
df$female <- as.numeric(df$sex == "Female")
df$male <- as.numeric(df$sex == "Male")
table(df$sex)

if (use_race == "race_nonwhite"){
  df$white <- as.numeric(df$race_nonwhite == "White")
  sense_cols <- append(sense_cols, "white")
  df$nonwhite <- as.numeric(df$race_nonwhite == "NonWhite")
  sense_cols <- append(sense_cols, "nonwhite")
  table(df$race_nonwhite)
}

if (use_race == "race_simpler"){
  df$white <- as.numeric(df$race_simpler == "White")
  sense_cols <- append(sense_cols, "white")
  df$black <- as.numeric(df$race_simpler == "Black")
  sense_cols <- append(sense_cols, "black")
  df$Latino <- as.numeric(df$race_simpler == "Latino")
  sense_cols <- append(sense_cols, "latino")
  df$asian <- as.numeric(df$race_simpler == "Asian")
  sense_cols <- append(sense_cols, "asian")
  df$other <- as.numeric(df$race_simpler == "Other")
  sense_cols <- append(sense_cols, "Other")
  table(df$race_simpler)
}

vars_m <- append(vars_m, sense_cols)


# Level 3 -----------------------------------------------------------------

# DAG: Sex -> UGPA; Race -> UGPA; Sex -> LSAT; Race -> LSAT
df_lev3 <- df

# Step 1: train model for descendant nodes, and 
model_ugpa <- lm(UGPA ~ female + nonwhite + 1, data=df_lev3)
summary(model_ugpa)

model_lsat <- lm(LSAT ~ female + nonwhite + 1, data=df_lev3)
# model_lsat <- lm(log(LSAT) ~ female + nonwhite + 1, data=df_lev3)
# exp(model_lsat$coefficients)
summary(model_lsat)


# perform the abduction step: estimate the residuals
df_lev3$resid_UGPA = df_lev3$UGPA - predict(model_ugpa, newdata=df_lev3)
hist(df_lev3$resid_UGPA)

df_lev3$resid_LSAT = df_lev3$LSAT - predict(model_lsat, newdata=df_lev3)
hist(df_lev3$resid_LSAT)

# Step 2: action on race and gender (accordingly: under multiple disc.)
# do(Gender:='Male')
df_lev3_do_male <- data.frame(female=rep(0, nrow(df_lev3)), 
                              nonwhite=df_lev3$nonwhite)

# do(Race:='White')
df_lev3_do_white <- data.frame(female=df_lev3$female, 
                               nonwhite=rep(0, nrow(df_lev3)))
# Step 3: prediction
# do(Gender:='Male')
df_lev3_do_male$Sex <- df_lev3$sex
df_lev3_do_male$Race <- df_lev3$race_nonwhite
df_lev3_do_male$resid_LSAT <- df_lev3$resid_LSAT
df_lev3_do_male$resid_UGPA <- df_lev3$resid_UGPA

df_lev3_do_male$scf_LSAT <- round(predict(model_lsat, newdata=df_lev3_do_male) 
                                  + df_lev3_do_male$resid_LSAT, 3)
df_lev3_do_male$scf_UGPA <- round(predict(model_ugpa, newdata=df_lev3_do_male) 
                                  + df_lev3_do_male$resid_UGPA, 3)

summary(df_lev3_do_male$scf_LSAT) # btw 10 - 48
summary(df_lev3_do_male$scf_UGPA) # btw 0 - 4

df_lev3_do_male$scf_LSAT <- 
  ifelse(df_lev3_do_male$scf_LSAT > 48.00, 48.00, df_lev3_do_male$scf_LSAT)
df_lev3_do_male$scf_LSAT <- 
  ifelse(df_lev3_do_male$scf_LSAT < 10.00, 10.00, df_lev3_do_male$scf_LSAT)
summary(df_lev3_do_male$scf_LSAT)

df_lev3_do_male$scf_UGPA <- 
  ifelse(df_lev3_do_male$scf_UGPA > 4.00, 4.00, df_lev3_do_male$scf_UGPA)
df_lev3_do_male$scf_UGPA <- 
  ifelse(df_lev3_do_male$scf_UGPA < 0.00, 0.00, df_lev3_do_male$scf_UGPA)
summary(df_lev3_do_male$scf_UGPA)

write.table(df_lev3_do_male, 
            file = paste(path_rslt, "cf_LawSchool_lev3_doMale.csv", sep = ""), 
            sep = "|")

# do(Race:='White')
df_lev3_do_white$Sex <- df_lev3$sex
df_lev3_do_white$Race <- df_lev3$race_nonwhite
df_lev3_do_white$resid_LSAT <- df_lev3$resid_LSAT
df_lev3_do_white$resid_UGPA <- df_lev3$resid_UGPA

df_lev3_do_white$scf_LSAT <- round(predict(model_lsat, newdata=df_lev3_do_white) 
                                   + df_lev3_do_white$resid_LSAT, 3)
df_lev3_do_white$scf_UGPA <- round(predict(model_ugpa, newdata=df_lev3_do_white) 
                                   + df_lev3_do_white$resid_UGPA, 3)

summary(df_lev3_do_white$scf_LSAT) # btw 10 - 48
summary(df_lev3_do_white$scf_UGPA) # btw 120 - 180

df_lev3_do_white$scf_LSAT <- 
  ifelse(df_lev3_do_white$scf_LSAT > 48.00, 48.00, df_lev3_do_white$scf_LSAT)
df_lev3_do_white$scf_LSAT <- 
  ifelse(df_lev3_do_white$scf_LSAT < 10.00, 10.00, df_lev3_do_white$scf_LSAT)
summary(df_lev3_do_white$scf_LSAT)

df_lev3_do_white$scf_UGPA <- 
  ifelse(df_lev3_do_white$scf_UGPA > 4.00, 4.00, df_lev3_do_white$scf_UGPA)
df_lev3_do_white$scf_UGPA <- 
  ifelse(df_lev3_do_white$scf_UGPA < 0.00, 0.00, df_lev3_do_white$scf_UGPA)
summary(df_lev3_do_white$scf_UGPA)

write.table(df_lev3_do_white, 
            file = paste(path_rslt, "cf_LawSchool_lev3_doWhite.csv", sep = ""), 
            sep = "|")

#
# EOF
#

# # Level 2 -----------------------------------------------------------------
# 
# # DAG: Sex -> UGPA; Race -> UGPA; Sex -> LSAT; Race -> LSAT; U -> UGPA; U -> LSAT
# df_lev2 <- df
# 
# #-- test on a sample
# trainIndex <- createDataPartition(df$sex, p = .5, list = FALSE, times = 1)
# df_lev2 <- df_lev2[trainIndex, ]
# 
# 
# # Step 1: Abduction via MCMC ----------------------------------------------
# 
# # Prepare data for Stan
# law_school_train <- list(N = nrow(df_lev2), K = length(sense_cols),
#                          a = data.matrix(df_lev2[ , sense_cols]),
#                          ugpa = df_lev2[ , c("UGPA")],
#                          lsat = df_lev2[ , c("LSAT")])
# 
# # Run the MCMC
# fit_law_school_train <- stan(file = paste(path_mdls, 'law_school_train.stan', sep=""),
#                              data = law_school_train,
#                              iter = 2000,
#                              chains = 1,
#                              verbose = TRUE)
# 
# # Store results (use load())
# save(fit_law_school_train, 
#      file = paste(path_data, "u_fit_law_school_train.RData", sep = ""))
# 
# # Extract the information
# la_law_school_train <- extract(fit_law_school_train, permuted=TRUE)
# 
# # Get (hyper)parameters
# U          <- colMeans(la_law_school_train$u)
# ugpa0      <- mean(la_law_school_train$ugpa0)
# eta_u_ugpa <- mean(la_law_school_train$eta_u_ugpa)
# eta_a_ugpa <- colMeans(la_law_school_train$eta_a_ugpa)
# lsat0      <- mean(la_law_school_train$lsat0)
# eta_u_lsat <- mean(la_law_school_train$eta_u_lsat)
# eta_a_lsat <- colMeans(la_law_school_train$eta_a_lsat)
# SIGMA_G    <- mean(la_law_school_train$sigma_g)
# 
# # store U's
# df_lev2$U <- U
# hist(U)
# 
# 
# # Steps 2 to 3 via MCMC estimates -----------------------------------------
# 
# # do(Gender='Male')
# do_male_lev2_opt1 <- df_lev2[ , c('sex', use_race)]
# do_male_lev2_opt1$U <- U
# 
# do_male_lev2_opt1$UGPA <- df_lev2$UGPA
# # predicted UGPA
# do_male_lev2_opt1$pr_UGPA <- 
#   ugpa0 + 
#   eta_u_ugpa*U + 
#   data.matrix(df_lev2[ , sense_cols])%*%eta_a_ugpa
# # cf UPGA
# do_male_lev2_opt1$scf_UGPA <- 
#   ugpa0 + 
#   eta_u_ugpa*U + 
#   data.matrix(data.frame(female=rep(0, nrow(df_lev2)), 
#                          male=rep(1, nrow(df_lev2)), 
#                          white=df_lev2$white, 
#                          nonwhite=df_lev2$nonwhite))%*%eta_a_ugpa
# # deltas (or 'error terms')
# hist(do_male_lev2_opt1$pr_UGPA - do_male_lev2_opt1$UGPA)
# 
# do_male_lev2_opt1$LSAT <- df_lev2$LSAT
# # predicted LSAT
# do_male_lev2_opt1$pr_LSAT <- exp(
#   lsat0 +
#   eta_u_lsat*U +
#   data.matrix(df_lev2[ , sense_cols])%*%eta_a_lsat)
# # cf LSAT
# do_male_lev2_opt1$scf_LSAT <- exp(
#   lsat0 +
#   eta_u_lsat*U + 
#   data.matrix(data.frame(female=rep(0, nrow(df_lev2)), 
#                          male=rep(1, nrow(df_lev2)), 
#                          white=df_lev2$white, 
#                          nonwhite=df_lev2$nonwhite))%*%eta_a_lsat)
# # deltas (or 'error terms')
# hist(do_male_lev2_opt1$pr_LSAT - do_male_lev2_opt1$LSAT)
# 
# summary(do_male_lev2_opt1$scf_UGPA)
# summary(do_male_lev2_opt1$scf_LSAT)
# 
# do_male_lev2_opt1$scf_LSAT <- 
#   ifelse(do_male_lev2_opt1$scf_LSAT > 48.00, 48.00, do_male_lev2_opt1$scf_LSAT)
# do_male_lev2_opt1$scf_LSAT <- 
#   ifelse(do_male_lev2_opt1$scf_LSAT < 10.00, 10.00, do_male_lev2_opt1$scf_LSAT)
# summary(do_male_lev2_opt1$scf_LSAT)
# 
# do_male_lev2_opt1$scf_UGPA <- 
#   ifelse(do_male_lev2_opt1$scf_UGPA > 4.00, 4.00, do_male_lev2_opt1$scf_UGPA)
# do_male_lev2_opt1$scf_UGPA <- 
#   ifelse(do_male_lev2_opt1$scf_UGPA < 0.00, 0.00, do_male_lev2_opt1$scf_UGPA)
# summary(do_male_lev2_opt1$scf_UGPA)
# 
# write.table(do_male_lev2_opt1, 
#             file = paste(path_rslt, "cf_LawSchool_lev2_1_doMale.csv", sep = ""), 
#             sep = "|")
# 
# # do(Race:='White')
# do_whites_lev2_opt1 <- df_lev2[ , c('sex', use_race)]
# do_whites_lev2_opt1$U <- U
# 
# do_whites_lev2_opt1$UGPA <- df_lev2$UGPA
# # predicted UGPA
# do_whites_lev2_opt1$pr_UGPA <-
#   ugpa0 +
#   eta_u_ugpa*U +
#   data.matrix(df_lev2[ , sense_cols])%*%eta_a_ugpa
# # cf UPGA
# do_whites_lev2_opt1$scf_UGPA <- 
#   ugpa0 + 
#   eta_u_ugpa*U + 
#   data.matrix(data.frame(female=df_lev2$female, 
#                          male=df_lev2$male, 
#                          white=rep(1, nrow(df_lev2)), 
#                          nonwhite=rep(0, nrow(df_lev2))))%*%eta_a_ugpa
# # deltas (or 'error terms')
# hist(do_whites_lev2_opt1$pr_UGPA - do_whites_lev2_opt1$UGPA)
# 
# do_whites_lev2_opt1$LSAT <- df_lev2$LSAT
# # predicted LSAT
# do_whites_lev2_opt1$pr_LSAT <- exp(
#   lsat0 +
#     eta_u_lsat*U +
#     data.matrix(df_lev2[ , sense_cols])%*%eta_a_lsat)
# # cf LSAT
# do_whites_lev2_opt1$scf_LSAT <- exp(
#   lsat0 +
#     eta_u_lsat*U + 
#     data.matrix(data.frame(female=df_lev2$female, 
#                            male=df_lev2$male, 
#                            white=rep(1, nrow(df_lev2)), 
#                            nonwhite=rep(0, nrow(df_lev2))))%*%eta_a_lsat)
# # deltas (or 'error terms')
# hist(do_whites_lev2_opt1$pr_LSAT - do_whites_lev2_opt1$LSAT)
# 
# summary(do_whites_lev2_opt1$scf_UGPA)
# summary(do_whites_lev2_opt1$scf_LSAT)
# 
# do_whites_lev2_opt1$scf_LSAT <- 
#   ifelse(do_whites_lev2_opt1$scf_LSAT > 48.00, 48.00, do_whites_lev2_opt1$scf_LSAT)
# do_whites_lev2_opt1$scf_LSAT <- 
#   ifelse(do_whites_lev2_opt1$scf_LSAT < 10.00, 10.00, do_whites_lev2_opt1$scf_LSAT)
# summary(do_whites_lev2_opt1$scf_LSAT)
# 
# do_whites_lev2_opt1$scf_UGPA <- 
#   ifelse(do_whites_lev2_opt1$scf_UGPA > 4.00, 4.00, do_whites_lev2_opt1$scf_UGPA)
# do_whites_lev2_opt1$scf_UGPA <- 
#   ifelse(do_whites_lev2_opt1$scf_UGPA < 0.00, 0.00, do_whites_lev2_opt1$scf_UGPA)
# summary(do_whites_lev2_opt1$scf_UGPA)
# 
# write.table(do_whites_lev2_opt1, 
#             file = paste(path_rslt, "cf_LawSchool_lev2_1_doWhite.csv", sep = ""), 
#             sep = "|")
# 
# 
# # Steps 1, 2, 3 given U (opt 2) -------------------------------------------
# 
# # After estimating U, we can incorporate it as an attribute for LSAT and UPGA
# # e.g., LSAT is a function of knowledge (i.e., U) and some randomness
# # Similar to opt 1, but trying to justify the MCMC step: the hidden confounder
# 
# # Step 1: train model for descendant nodes, and 
# model_ugpa_lev2 <- lm(UGPA ~
#                         female + nonwhite + U + 1,
#                       data=df_lev2)
# 
# model_lsat_lev2 <- lm(LSAT ~
#                         female + nonwhite + U + 1,
#                       data=df_lev2)
# 
# # perform the abduction step: estimate the residuals
# df_lev2$resid_UGPA = df_lev2$UGPA - 
#   predict.glm(model_ugpa_lev2, newdata=df_lev2)
# hist(df_lev2$resid_UGPA)
# 
# df_lev2$resid_LSAT = df_lev2$LSAT - 
#   predict.glm(model_lsat_lev2, newdata=df_lev2)
# hist(df_lev2$resid_LSAT)
# 
# # Step 2: action on race and gender (accordingly: under multiple disc.)
# # do(Gender:='Male')
# df_lev2_do_male <- data.frame(female=rep(0, nrow(df_lev2)),
#                               nonwhite=df_lev2$nonwhite)
# 
# # do(Race:='White')
# df_lev2_do_white <- data.frame(female=df_lev2$female,
#                                nonwhite=rep(0, nrow(df_lev2)))
# # Step 3: prediction
# # do(Gender:='Male')
# df_lev2_do_male$Sex <- df_lev2$sex
# df_lev2_do_male$Race <- df_lev2$race_nonwhite
# df_lev2_do_male$resid_LSAT <- df_lev2$resid_LSAT
# df_lev2_do_male$resid_UGPA <- df_lev2$resid_UGPA
# 
# df_lev2_do_male$scf_LSAT <- round(
#   predict.glm(model_lsat_lev2, newdata=df_lev2_do_male) + df_lev2_do_male$resid_LSAT,3
#   )
# df_lev2_do_male$scf_UGPA <- round(
#   predict.glm(model_ugpa_lev2, newdata=df_lev2_do_male) + df_lev2_do_male$resid_UGPA, 3
#   )
# 
# summary(df_lev2_do_male$scf_LSAT) # btw 10 - 48
# summary(df_lev2_do_male$scf_UGPA) # btw 0 - 4
# 
# df_lev2_do_male$scf_LSAT <- 
#   ifelse(df_lev2_do_male$scf_LSAT > 48.00, 48.00, df_lev2_do_male$scf_LSAT)
# df_lev2_do_male$scf_LSAT <- 
#   ifelse(df_lev2_do_male$scf_LSAT < 10.00, 10.00, df_lev2_do_male$scf_LSAT)
# summary(df_lev2_do_male$scf_LSAT)
# 
# df_lev2_do_male$scf_UGPA <- 
#   ifelse(df_lev2_do_male$scf_UGPA > 4.00, 4.00, df_lev2_do_male$scf_UGPA)
# df_lev2_do_male$scf_UGPA <- 
#   ifelse(df_lev2_do_male$scf_UGPA < 0.00, 0.00, df_lev2_do_male$scf_UGPA)
# summary(df_lev2_do_male$scf_UGPA)
# 
# write.table(df_lev2_do_male, 
#             file = paste(path_rslt, "cf_LawSchool_lev2_2_doMale.csv", sep = ""), 
#             sep = "|")
# 
# # do(Race:='White')
# df_lev2_do_white$Sex <- df_lev2$sex
# df_lev2_do_white$Race <- df_lev2$race_nonwhite
# df_lev2_do_white$resid_LSAT <- df_lev2$resid_LSAT
# df_lev2_do_white$resid_UGPA <- df_lev2$resid_UGPA
# 
# df_lev2_do_white$scf_LSAT <- round(
#   predict.glm(model_lsat_lev2, newdata=df_lev2_do_white) + df_lev2_do_white$resid_LSAT, 3
#   )
# 
# df_lev2_do_white$scf_UGPA <- round(
#   predict(model_ugpa_lev2, newdata=df_lev2_do_white) + df_lev2_do_white$resid_UGPA, 3
#   )
# 
# summary(df_lev2_do_white$scf_LSAT) # btw 10 - 48
# summary(df_lev2_do_white$scf_UGPA) # btw 120 - 180
# 
# df_lev2_do_white$scf_LSAT <- 
#   ifelse(df_lev2_do_white$scf_LSAT > 48.00, 48.00, df_lev2_do_white$scf_LSAT)
# df_lev2_do_white$scf_LSAT <- 
#   ifelse(df_lev2_do_white$scf_LSAT < 10.00, 10.00, df_lev2_do_white$scf_LSAT)
# summary(df_lev2_do_white$scf_LSAT)
# 
# df_lev2_do_white$scf_UGPA <- 
#   ifelse(df_lev2_do_white$scf_UGPA > 4.00, 4.00, df_lev2_do_white$scf_UGPA)
# df_lev2_do_white$scf_UGPA <- 
#   ifelse(df_lev2_do_white$scf_UGPA < 0.00, 0.00, df_lev2_do_white$scf_UGPA)
# summary(df_lev2_do_white$scf_UGPA)
# 
# write.table(df_lev2_do_white, 
#             file = paste(path_rslt, "cf_LawSchool_lev2_2_doWhite.csv", sep = ""), 
#             sep = "|")
