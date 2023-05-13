library(tidymodels)
library(modeldata)
library(glmnet)

# Load data
data(diamonds)

for(i in 1:5) {
  rand_var_name <- sample(LETTERS, 5) %>% paste(collapse = '')
  diamonds[rand_var_name] <- runif(n = nrow(diamonds), min = sample(-10:-1, 1), max = sample(1:10, 1))
}

diamonds <- mutate(
  diamonds, 
  cut = case_when(
    cut %in% c('Ideal', 'Premium') ~ 1,
    TRUE ~ 0
  ) %>% forcats::as_factor()
)

set.seed(2369)
tr_te_split <- initial_split(diamonds, prop = 3/4)
dia_train <- training(tr_te_split)
dia_test  <- testing(tr_te_split)

set.seed(1697)
folds <- vfold_cv(dia_train, v = 10, strata = cut)

dia_pre_proc <-
  recipe(cut ~ ., data = dia_train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

glm_mod <-
  parsnip::logistic_reg(mode = 'classification', engine = 'glm', penalty = tune(), mixture = tune()) %>%
  set_engine('glmnet')

glm_wflow <-
  workflow() %>%
  add_model(glm_mod) %>%
  add_recipe(dia_pre_proc)


set.seed(12)
search_res <-
  glm_wflow %>% 
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 25,
    metrics = metric_set(roc_auc),
    control = control_bayes(no_improve = 5, verbose = TRUE)
  )



best_wflow <- select_best(search_res)
final_wflow <- finalize_workflow(glm_wflow, best)
fit_wflow <- fit(final_wflow, data = dia_train)
predicted <- predict(fit_wflow, dia_test)

butchered_wflow <- butcher::axe_data(fit_wflow)

saveRDS(fit_wflow, glue::glue('./results/fit_wflow_{Sys.Date()}.RDS'))
saveRDS(butchered_wflow, glue::glue('./results/butchered_wflow_{Sys.Date()}.RDS'))