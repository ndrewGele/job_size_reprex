library(tidymodels)
library(glmnet)

cl <- parallelly::availableCores() %>% 
  `*`(.5) %>% 
  parallelly::makeClusterPSOCK()

doParallel::registerDoParallel(cl)

# Load data
data(diamonds)

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
folds <- vfold_cv(dia_train, v = 5, strata = cut)

dia_pre_proc <-
  recipe(cut ~ ., data = dia_train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

glm_mod <-
  parsnip::logistic_reg(mode = 'classification', penalty = tune(), mixture = tune()) %>%
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
    initial = 5,
    iter = 30,
    metrics = metric_set(roc_auc),
    control = control_bayes(no_improve = 10, verbose = TRUE)
  )

doParallel::stopImplicitCluster()

best_wflow <- select_best(search_res, metric = 'roc_auc')
final_wflow <- finalize_workflow(glm_wflow, best_wflow)
fit_wflow <- fit(final_wflow, data = dia_train)
butchered_wflow <- butcher::axe_data(fit_wflow)

dttm <- format(Sys.time(), '%Y_%m_%d_%H_%M_%S')
new_dir <- glue::glue('./results_{dttm}')
if(!dir.exists(new_dir)) dir.create(new_dir)

saveRDS(diamonds, glue::glue('./{new_dir}/diamonds.RDS'))
saveRDS(fit_wflow, glue::glue('./{new_dir}/fit_wflow.RDS'))
saveRDS(butchered_wflow, glue::glue('./{new_dir}/butchered_wflow.RDS'))

size <- lapply(list.files(new_dir, full.names = TRUE), file.size) %>% 
  unlist() %>% 
  sum()

file.create(file.path(new_dir, paste0(size, '.txt')))
