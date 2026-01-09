library(tidyverse)
library(haven)
library(catboost)
library(caret)
library(parallel)

set.seed(123)

# ===============================================================
# VERSION 4 (with SMALL GRID SEARCH)
#   - Target: q8a (q8a_rec dropped)
#   - Treat q8a == 9 as NA (DK/NA not a valid bin)
#   - Use CatBoost MultiClass to get probabilities
#   - Small grid search ONCE on a time-aware wave holdout (diagnostic only)
#   - Fit final model on ALL observed q8a
#   - Impute missing q8a via PROBABILITIES (stochastic draw)
#   - No bin-calibration; avoid artificial breaks by using full observed fit
# ===============================================================


# ===============================================================
# BLOCK A: Data cleaning  (MINIMAL CHANGES)  --- REWRITTEN
#   - Keep your filters (waves 31/33/35 excluded here)
#   - Create NA for q8a when:
#       (a) coded as 9 (DK/NA)
#       (b) empty / blank string
#       (c) genuine NA in the raw file
#   - Remove the redundant "force NA for waves 31/33/35" line
# ===============================================================

dataset_A <- read_dta("C:/Financing Gap/ecb.SAFE_microdata/safepanel_allrounds.dta") %>%
  filter(
    d0 %in% c("BE","DK","DE","EE","IE","GR","ES","FR","HR","IT",
              "CY","LV","LT","LU","MT","NL","AT","PL","PT","RO",
              "SI","SK","FI","SE","BG","CZ","HU"),
    wave > 10,
    !wave %in% c(31, 33, 35),
    q7a_a %in% c(1, 2)                                  # demanders filter
  ) %>%
  select(
    -c(wgtcommon, wgtentr, wgtoldentr, wgtoldcommon,
       ida, experiment_1, experiment_2, intdate),
    -any_of("q8a_rec")                                  # drop q8a_rec
  ) %>%
  arrange(permid, wave) %>%
  mutate(
    across(
      -c(permid, wave),
      ~ if (inherits(.x, "haven_labelled")) haven::as_factor(.x) else as.factor(.x)
    ),
    # ---- q8a cleaning: map DK/NA=9 and blank/empty answers to NA ----
    q8a = as.character(q8a),
    q8a = stringr::str_trim(q8a),
    q8a = dplyr::na_if(q8a, ""),                         # empty string -> NA
    q8a = stringr::str_extract(q8a, "\\d+"),             # keep first numeric code
    q8a = ifelse(q8a == "9", NA_character_, q8a),        # DK/NA (9) -> NA
    q8a = factor(q8a)
  )

cat("Step 1 completed | Rows:", nrow(dataset_A),
    "| Columns:", ncol(dataset_A),
    "| Missing q8a:", sum(is.na(dataset_A$q8a)), "\n")

# ---------------------------------------------------------------
# A.2 Missingness target dependence test (Option C) (unchanged)
# ---------------------------------------------------------------

missingness_informative <- function(data, var, target, p_thresh = 0.20) {
  keep_rows <- !is.na(data[[target]])
  d <- data[keep_rows, , drop = FALSE]
  
  miss_flag <- is.na(d[[var]])
  
  if (all(!miss_flag)) return(TRUE)
  
  tab <- table(miss_flag, d[[target]])
  
  if (min(dim(tab)) < 2) return(TRUE)
  
  pval <- suppressWarnings(chisq.test(tab)$p.value)
  
  !is.na(pval) && pval < p_thresh
}


# ---------------------------------------------------------------
# A.3 Apply the hybrid rule
#   1) Drop variables with >90% missing
#   2) Keep variables whose missingness predicts TARGET (q8a)
#   3) Explicit NA encoding for factor predictors, excluding q8a
#   + Force-keep q7a_a
# ---------------------------------------------------------------

id_vars     <- c("permid", "wave", "d0")
target_var  <- "q8a"
leak_vars   <- c("q8a")

# 1) Drop almost-empty variables
hard_drop <- names(which(colMeans(is.na(dataset_A)) > 0.90))
dataset_A  <- dataset_A %>% select(-all_of(hard_drop))

# 2) Option C selection (tested only where q8a is observed)
candidate_vars <- setdiff(names(dataset_A), c(id_vars, leak_vars))

vars_to_keep <- candidate_vars[
  sapply(candidate_vars, function(v)
    missingness_informative(dataset_A, v, target_var))
]

# force keep q7a_a if present
vars_to_keep <- union(vars_to_keep, intersect("q7a_a", names(dataset_A)))

# Keep id + target + selected predictors
dataset_A <- dataset_A %>%
  select(all_of(c(id_vars, target_var, vars_to_keep)))

# 3) Explicit NA encoding for predictors ONLY (exclude q8a)
pred_cols <- setdiff(names(dataset_A), c(id_vars, leak_vars))
dataset_A <- dataset_A %>%
  mutate(across(all_of(pred_cols),
                ~ if (is.factor(.x)) forcats::fct_explicit_na(.x, "MISSING") else .x))

cat("Step 2 completed | Columns after missingness control:",
    ncol(dataset_A), "\n")


# ===============================================================
# BLOCK B: Model table preparation
#   - Prepare observed-q8a rows for model training
#   - Correct order used for ordinal diagnostics: 1 < 2 < 5 < 6 < 4
#   - CatBoost target for MultiClass must be 0..K-1
# ===============================================================

model_df <- dataset_A %>% filter(!is.na(.data[[target_var]]))

y_factor <- as.factor(model_df[[target_var]])

# correct order for your bins (used for metrics + ordered factor labels)
ord_levels   <- intersect(c("1","2","5","6","4"), levels(y_factor))
y_ord_factor <- factor(y_factor, levels = ord_levels, ordered = TRUE)

y_levels  <- levels(y_ord_factor)
n_classes <- length(y_levels)

# ordinal ints for metrics: 1..K
y_ord <- as.integer(y_ord_factor)

# class labels for CatBoost MultiClass: 0..K-1
y_cls <- y_ord - 1L

# Predictors: exclude permid + q8a (prevents leakage)
X_all <- model_df %>% select(-all_of(c("permid", leak_vars)))

cat_features <- which(
  sapply(X_all, function(x) is.factor(x) || is.character(x))
) - 1L

# Use MultiClass to get probabilities
loss_fun <- "MultiClass"


# ===============================================================
# BLOCK C: Time-aware split by wave (DIAGNOSTIC holdout for tuning)
#   - This is used ONLY to pick hyperparameters once
# ===============================================================

uniq_waves <- sort(unique(model_df$wave))
test_n     <- max(1L, floor(0.20 * length(uniq_waves)))   # small holdout by wave
test_waves <- tail(uniq_waves, test_n)

train_idx <- which(!model_df$wave %in% test_waves)
test_idx  <- which( model_df$wave %in% test_waves)

# Targets for diagnostics/metrics (ordinal)
y_train <- y_ord[train_idx]
y_test  <- y_ord[test_idx]

# Targets for CatBoost (class 0..K-1)
y_train_cls <- y_cls[train_idx]
y_test_cls  <- y_cls[test_idx]

# Optional: inverse-frequency weights (same spirit as V3)
freq_train <- table(y_train_cls)
w_map      <- median(as.numeric(freq_train)) / as.numeric(freq_train)
names(w_map) <- names(freq_train)
w_train <- as.numeric(w_map[as.character(y_train_cls)])
w_train[is.na(w_train)] <- 1

train_pool <- catboost.load_pool(
  X_all[train_idx, ], y_train_cls,
  cat_features = cat_features,
  weight       = w_train
)

test_pool <- catboost.load_pool(
  X_all[test_idx, ], y_test_cls,
  cat_features = cat_features
)


# ===============================================================
# BLOCK C2: Small Grid Search (ONCE)
#   - Evaluate by MultiClass logloss on test_pool
#   - Early stopping on test_pool to pick best_iter per run
# ===============================================================

grid <- expand.grid(
  depth         = c(5, 6, 8),
  learning_rate = c(0.03, 0.05),
  l2_leaf_reg   = c(3, 5)
)

results <- vector("list", nrow(grid))

safe_logloss <- function(probs, y_true_cls) {
  # probs: N x K, y_true_cls: 0..K-1
  eps <- 1e-15
  if (is.null(dim(probs))) probs <- matrix(probs, ncol = n_classes, byrow = TRUE)
  idx <- cbind(seq_len(nrow(probs)), y_true_cls + 1L)
  p_true <- pmax(probs[idx], eps)
  -mean(log(p_true))
}

for (i in seq_len(nrow(grid))) {
  
  params <- list(
    loss_function  = loss_fun,
    eval_metric    = "MultiClass",
    iterations     = 100,
    learning_rate  = grid$learning_rate[i],
    depth          = grid$depth[i],
    l2_leaf_reg    = grid$l2_leaf_reg[i],
    od_type        = "Iter",
    od_wait        = 80,
    random_seed    = 123,
    thread_count   = detectCores(),
    use_best_model = TRUE,
    logging_level  = "Silent"
  )
  
  m <- catboost.train(train_pool, test_pool, params = params)
  
  probs_i <- catboost.predict(m, test_pool, prediction_type = "Probability")
  logloss_i <- safe_logloss(probs_i, y_test_cls)
  
  best_iter_i <- tryCatch(catboost.get_best_iteration(m), error = function(e) NA_integer_)
  if (!is.na(best_iter_i)) best_iter_i <- best_iter_i + 1L
  if (is.na(best_iter_i))  best_iter_i <- params$iterations
  
  results[[i]] <- list(
    params    = grid[i, ],
    best_iter = best_iter_i,
    logloss   = logloss_i
  )
  
  cat(sprintf("Grid %02d/%02d | Logloss %.5f | Iter %d | depth=%d lr=%.3f l2=%d\n",
              i, nrow(grid),
              logloss_i, best_iter_i,
              grid$depth[i], grid$learning_rate[i], grid$l2_leaf_reg[i]))
}

best_id <- which.min(sapply(results, `[[`, "logloss"))
best    <- results[[best_id]]

cat("\nBest grid choice:\n")
print(best$params)
cat(sprintf("Best holdout logloss: %.5f | Best iteration: %d\n", best$logloss, best$best_iter))


# ===============================================================
# BLOCK D: Final Model (fit on ALL observed q8a)
#   - Uses best hyperparameters + best_iter from the ONE-SHOT grid
# ===============================================================

# Weights on ALL observed (keeps the same training philosophy)
freq_all <- table(y_cls)
w_map_all <- median(as.numeric(freq_all)) / as.numeric(freq_all)
names(w_map_all) <- names(freq_all)
w_all <- as.numeric(w_map_all[as.character(y_cls)])
w_all[is.na(w_all)] <- 1

all_pool <- catboost.load_pool(
  X_all, y_cls,
  cat_features = cat_features,
  weight       = w_all
)

final_params <- list(
  loss_function  = loss_fun,
  eval_metric    = "MultiClass",
  iterations     = best$best_iter,
  learning_rate  = best$params$learning_rate,
  depth          = best$params$depth,
  l2_leaf_reg    = best$params$l2_leaf_reg,
  random_seed    = 123,
  thread_count   = detectCores(),
  logging_level  = "Silent"
)

final_model <- catboost.train(all_pool, params = final_params)


# ===============================================================
# BLOCK E: Diagnostic Metrics (Ordinal-style, on holdout waves)
#   - This is for sanity checks only (NOT the final fit target)
#   - We refit a diagnostic model on train_pool with best params to score test_pool
# ===============================================================

diag_params <- final_params
diag_params$iterations     <- 500
diag_params$use_best_model <- TRUE
diag_params$od_type        <- "Iter"
diag_params$od_wait        <- 80

diag_model <- catboost.train(train_pool, test_pool, params = diag_params)

probs_test <- catboost.predict(diag_model, test_pool, prediction_type = "Probability")
if (is.null(dim(probs_test))) probs_test <- matrix(probs_test, ncol = n_classes, byrow = TRUE)

# argmax class (0..K-1) -> ordinal int (1..K)
pred_cls  <- max.col(probs_test) - 1L
pred_int  <- pred_cls + 1L

# expected value on ordinal scale 1..K (for RMSE/MAE/Spearman)
pred_ev <- as.vector(probs_test %*% (1:n_classes))

rmse_test <- sqrt(mean((pred_ev - y_test)^2))
mae_test  <- mean(abs(pred_ev - y_test))
spearman  <- suppressWarnings(cor(pred_ev, y_test, method = "spearman"))

acc_exact   <- mean(pred_int == y_test)
acc_within1 <- mean(abs(pred_int - y_test) <= 1L)

pred_factor <- factor(pred_int, levels = 1:n_classes, labels = y_levels, ordered = TRUE)
obs_factor  <- factor(y_test,    levels = 1:n_classes, labels = y_levels, ordered = TRUE)

cm <- confusionMatrix(pred_factor, obs_factor)

quadratic_weighted_kappa <- function(truth_int, pred_int, k) {
  O <- table(factor(truth_int, levels = 1:k),
             factor(pred_int,  levels = 1:k))
  O <- as.matrix(O)
  N <- sum(O)
  if (N == 0) return(NA_real_)
  
  row_m <- rowSums(O)
  col_m <- colSums(O)
  E <- outer(row_m, col_m) / N
  
  W <- outer(1:k, 1:k, function(i, j) ((i - j)^2) / ((k - 1)^2))
  
  1 - (sum(W * O) / sum(W * E))
}

qwk <- quadratic_weighted_kappa(y_test, pred_int, n_classes)

macro_f1 <- if (is.matrix(cm$byClass)) {
  mean(cm$byClass[, "F1"], na.rm = TRUE)
} else {
  unname(cm$byClass["F1"])
}

cat("\n================ DIAGNOSTIC METRICS (ORDINAL) ================\n")
cat(sprintf("Target                 : %s\n", target_var))
cat(sprintf("RMSE (EV)              : %.4f\n", rmse_test))
cat(sprintf("MAE  (EV)              : %.4f\n", mae_test))
cat(sprintf("Spearman rho (EV)      : %.4f\n", spearman))
cat(sprintf("Exact accuracy         : %.4f\n", acc_exact))
cat(sprintf("Within-1 accuracy      : %.4f\n", acc_within1))
cat(sprintf("Quadratic weighted kappa: %.4f\n\n", qwk))
cat(sprintf("Macro F1               : %.4f\n", macro_f1))

cat("---- Confusion matrix (holdout waves) ----\n")
print(cm$table)


# ===============================================================
# BLOCK E2: Credibility diagnostics (holdout waves, same as V3 style)
# ===============================================================

eval_df <- model_df[test_idx, ] %>%
  mutate(
    q8a_true_int = y_test,
    q8a_pred_int = pred_int,
    q8a_true_bin = factor(y_test,   levels = 1:n_classes, labels = y_levels, ordered = TRUE),
    q8a_pred_bin = factor(pred_int, levels = 1:n_classes, labels = y_levels, ordered = TRUE)
  )

make_dist <- function(df, group_vars = NULL) {
  
  if (is.null(group_vars) || length(group_vars) == 0) {
    df <- df %>% mutate(grp = "ALL")   # Stata-safe name
    group_vars <- "grp"
  }
  
  ct_true <- df %>%
    count(across(all_of(group_vars)), q8a_true_bin, name = "n_true") %>%
    rename(bin = q8a_true_bin)
  
  ct_pred <- df %>%
    count(across(all_of(group_vars)), q8a_pred_bin, name = "n_pred") %>%
    rename(bin = q8a_pred_bin)
  
  grid2 <- tidyr::crossing(
    df %>% distinct(across(all_of(group_vars))),
    bin = factor(y_levels, levels = y_levels, ordered = TRUE)
  )
  
  dist <- grid2 %>%
    left_join(ct_true, by = c(group_vars, "bin")) %>%
    left_join(ct_pred, by = c(group_vars, "bin")) %>%
    mutate(
      n_true = tidyr::replace_na(n_true, 0L),
      n_pred = tidyr::replace_na(n_pred, 0L)
    ) %>%
    group_by(across(all_of(group_vars))) %>%
    mutate(
      share_true = n_true / sum(n_true),
      share_pred = n_pred / sum(n_pred),
      diff_share = share_pred - share_true
    ) %>%
    ungroup()
  
  tvd_tbl <- dist %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(
      n = sum(n_true),
      tvd = 0.5 * sum(abs(share_pred - share_true)),
      .groups = "drop"
    ) %>%
    left_join(
      df %>%
        group_by(across(all_of(group_vars))) %>%
        summarise(
          mean_true_int = mean(q8a_true_int, na.rm = TRUE),
          mean_pred_int = mean(q8a_pred_int, na.rm = TRUE),
          mean_shift    = mean_pred_int - mean_true_int,
          .groups = "drop"
        ),
      by = group_vars
    )
  
  list(dist = dist, tvd = tvd_tbl)
}

overall    <- make_dist(eval_df)
by_wave    <- make_dist(eval_df, c("wave"))
by_country <- make_dist(eval_df, c("d0"))

dist_all <- bind_rows(
  overall$dist    %>% mutate(scope = "overall", wave = NA_real_, d0 = NA_character_) %>% select(scope, wave, d0, everything()),
  by_wave$dist    %>% mutate(scope = "wave",   d0 = NA_character_)                  %>% select(scope, wave, d0, everything()),
  by_country$dist %>% mutate(scope = "d0",     wave = NA_real_)                     %>% select(scope, wave, d0, everything())
)

tvd_all <- bind_rows(
  overall$tvd    %>% mutate(scope = "overall", wave = NA_real_, d0 = NA_character_) %>% select(scope, wave, d0, everything()),
  by_wave$tvd    %>% mutate(scope = "wave",   d0 = NA_character_)                   %>% select(scope, wave, d0, everything()),
  by_country$tvd %>% mutate(scope = "d0",     wave = NA_real_)                      %>% select(scope, wave, d0, everything())
)

haven::write_dta(dist_all, "C:/Financing Gap/Comparison Metrics/Method_B_Diag_dist_V4.dta")
haven::write_dta(tvd_all,  "C:/Financing Gap/Comparison Metrics/Method_B_Diag_tvd_V4.dta")

cat("\n================ CREDIBILITY DIAGNOSTICS (SHORT) ================\n")
cat(sprintf("Overall TVD: %.4f\n", overall$tvd$tvd[1]))
cat(sprintf("Avg TVD by wave: %.4f | Avg TVD by country: %.4f\n",
            mean(by_wave$tvd$tvd, na.rm = TRUE),
            mean(by_country$tvd$tvd, na.rm = TRUE)))

worst_waves <- by_wave$tvd %>% arrange(desc(tvd)) %>% head(10)
worst_d0    <- by_country$tvd %>% arrange(desc(tvd)) %>% head(10)

cat("\nWorst 10 waves by TVD:\n"); print(worst_waves)
cat("\nWorst 10 countries by TVD:\n"); print(worst_d0)


# ===============================================================
# BLOCK F: Probabilistic imputation (FINAL DATASET)  --- TWO COLUMNS
#   - q8a         : raw (observed + NA where missing)
#   - q8a_imputed : observed where available, otherwise imputed draw
#   - q8a_imp_flag: 1 if imputed, 0 if observed
# ===============================================================

# 0) Create q8a_imputed as a copy of the raw q8a
dataset_A$q8a_imputed <- dataset_A$q8a

q8a_was_missing <- is.na(dataset_A$q8a_imputed)
miss_idx <- which(q8a_was_missing)

# IMPORTANT: lock the feature set to exactly what the model was trained on
feature_names <- names(X_all)   # X_all from Block B

if (length(miss_idx) > 0) {
  
  X_miss <- dataset_A[miss_idx, feature_names, drop = FALSE]
  
  miss_pool <- catboost.load_pool(
    X_miss,
    cat_features = cat_features
  )
  
  probs_miss <- catboost.predict(final_model, miss_pool, prediction_type = "Probability")
  if (is.null(dim(probs_miss))) probs_miss <- matrix(probs_miss, ncol = n_classes, byrow = TRUE)
  
  set.seed(123)
  
  draw_one <- function(p) sample(y_levels, size = 1, prob = p)
  
  q8a_draws <- apply(probs_miss, 1, draw_one)
  
  dataset_A$q8a_imputed[miss_idx] <- q8a_draws
}

# enforce consistent ordered factor levels on BOTH columns
dataset_A$q8a <- factor(as.character(dataset_A$q8a), levels = y_levels, ordered = TRUE)

dataset_A$q8a_imputed <- factor(as.character(dataset_A$q8a_imputed),
                                levels = y_levels, ordered = TRUE)

# transparency flag (1 = imputed, 0 = observed)
dataset_A$q8a_imp_flag <- ifelse(q8a_was_missing, 1L, 0L)

final_output <- dataset_A

haven::write_dta(final_output, "C:/Financing Gap/Dataset_Method_B_V4_q8a.dta")
cat("\nSaved: C:/Financing Gap/Dataset_Method_B_V4_q8a.dta\n")

# ===============================================================
# BLOCK G: MERGER WITH THE NON IMPUTED DATASET
# ===============================================================

library(stringr)

# -----------------------------
# 0) Helpers
# -----------------------------
numish <- function(x) as.numeric(as.character(x))

# This is the factor-level order you used in V4:
# y_levels came from ord_levels = c("1","2","5","6","4")
lvl_to_safe_code <- c(1, 2, 5, 6, 4)  # index 1..5 -> SAFE code

# -----------------------------
# 1) Load ORIGINAL full dataset (no filters) and normalize keys
# -----------------------------
orig <- read_dta("C:/Financing Gap/ecb.SAFE_microdata/safepanel_allrounds.dta") %>%
  as_tibble() %>%
  mutate(
    permid = as.numeric(permid),
    wave   = as.numeric(wave),
    d0     = str_trim(as.character(d0))
  )

# Map (permid,wave) -> true d0, to fix V4's numeric-coded d0
orig_pw_d0 <- orig %>%
  select(permid, wave, d0) %>%
  distinct(permid, wave, d0)

# -----------------------------
# 2) Load V4 output (demanders-only) and recover SAFE codes for q8a_imputed
# -----------------------------
v4 <- read_dta("C:/Financing Gap/Dataset_Method_B_V4_q8a.dta") %>%
  as_tibble() %>%
  mutate(
    permid = as.numeric(permid),
    wave   = as.numeric(wave),
    
    # V4 q8a_imputed is likely stored as 1..5 (factor index)
    q8a_imp_raw = numish(q8a_imputed),
    
    # Convert to actual SAFE code using your known level order
    q8a_imp_safe = dplyr::case_when(
      is.na(q8a_imp_raw) ~ NA_real_,
      q8a_imp_raw %in% 1:5 ~ as.numeric(lvl_to_safe_code[q8a_imp_raw]),
      q8a_imp_raw %in% c(1,2,4,5,6,9) ~ q8a_imp_raw,   # fallback if it’s already SAFE-coded
      TRUE ~ NA_real_
    )
  ) %>%
  select(permid, wave, q8a_imp_safe) %>%
  distinct(permid, wave, .keep_all = TRUE)

# Recover true d0 for V4 rows from orig
imp <- v4 %>%
  left_join(orig_pw_d0, by = c("permid","wave")) %>%
  transmute(permid, wave, d0, q8a_imp_safe) %>%
  distinct(permid, wave, d0, .keep_all = TRUE)

# Hard check: all V4 rows found d0 in orig
stopifnot(sum(is.na(imp$d0)) == 0)

# -----------------------------
# 3) Merge onto full panel and create EXACTLY the two columns you want
# -----------------------------
merged2 <- orig %>%
  left_join(imp, by = c("permid","wave","d0")) %>%
  mutate(
    # q8a_imputed = original q8a, but replace where V4 provides a value
    q8a_imputed = if_else(!is.na(q8a_imp_safe), q8a_imp_safe, numish(q8a))
  ) %>%
  select(-q8a_imp_safe)   # drop helper

# Save
write_dta(merged2, "C:/Financing Gap/Dataset_Method_B_V4_q8a_merged.dta")
cat("Saved:\nC:/Financing Gap/Dataset_Method_B_V4_q8a_merged.dta\n")

### TEST

# q8a unchanged?
stopifnot(all.equal(numish(orig$q8a), numish(merged2$q8a), check.attributes = FALSE))

# how many got replaced by V4?
n_replaced <- sum(!is.na(merged2$q8a_imputed) & is.na(numish(orig$q8a)) & !is.na(merged2$q8a_imputed))
cat("Rows newly filled (orig missing, now filled):", n_replaced, "\n")

# Demanders subset count (should equal your V4 rows, after d0 mapping)
cat("Rows where V4 was available:", sum(!is.na(merged2$q8a_imputed) & !is.na(merged2$permid)), "\n")
