#' xgboost: eXtreme Gradient Boosting
#'
#' This learner provides fitting procedures for \code{xgboost} models, using
#' the \pkg{xgboost} package, via \code{\link[xgboost]{xgb.train}}. Such
#' models are classification and regression trees with extreme gradient
#' boosting. For details on the fitting procedure, consult the documentation of
#' the \pkg{xgboost} and \insertCite{xgboost;textual}{sl3}).
#'
#' @docType class
#'
#' @importFrom R6 R6Class
#' @importFrom stats predict
#'
#' @export
#'
#' @keywords data
#'
#' @return A learner object inheriting from \code{\link{Lrnr_base}} with
#'  methods for training and prediction. For a full list of learner
#'  functionality, see the complete documentation of \code{\link{Lrnr_base}}.
#'
#' @format An \code{\link[R6]{R6Class}} object inheriting from
#'  \code{\link{Lrnr_base}}.
#'
#' @family Learners
#'
#' @seealso [Lrnr_gbm] for standard gradient boosting models (via the \pkg{gbm}
#'  package) and [Lrnr_lightgbm] for the faster and more efficient gradient
#'  boosted trees from the LightGBM framework (via the \pkg{lightgbm} package).
#'
#' @section Parameters:
#'   - \code{nrounds=20}: Number of fitting iterations.
#'   - \code{...}: Other parameters passed to \code{\link[xgboost]{xgb.train}}.
#'
#' @references
#'  \insertAllCited{}
#'
#' @examples
#' data(mtcars)
#' mtcars_task <- sl3_Task$new(
#'   data = mtcars,
#'   covariates = c(
#'     "cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am",
#'     "gear", "carb"
#'   ),
#'   outcome = "mpg"
#' )
#'
#' # initialization, training, and prediction with the defaults
#' xgb_lrnr <- Lrnr_xgboost$new()
#' xgb_fit <- xgb_lrnr$train(mtcars_task)
#' xgb_preds <- xgb_fit$predict()
#'
#' # get feature importance from fitted model
#' xgb_varimp <- xgb_fit$importance()
Lrnr_xgboost <- R6::R6Class(
  classname = "Lrnr_xgboost", inherit = Lrnr_base,
  public = list(
    initialize = function(nrounds = 20, nthread = 1, ...) {
      params <- args_to_list()
      super$initialize(params = params, ...)
    },
    importance = function(...) {
      self$assert_trained()
      args <- list(...)
      args$model <- self$fit_object
      imp <- call_with_args(xgboost::xgb.importance, args)
      rownames(imp) <- imp[["Feature"]]
      imp
    }
  ),
  private = list(
    .properties = c("continuous", "binomial", "categorical", "weights", "offset", "importance"),

    .audit_df = function(df, head_levels = 4L) {
      cls <- vapply(df, function(z) paste(class(z), collapse = ","), character(1))
      is_char <- vapply(df, is.character, logical(1))
      is_fac  <- vapply(df, is.factor,   logical(1))
      lvl_n   <- vapply(df, function(z) if (is.factor(z)) length(levels(z)) else NA_integer_, integer(1))
      lvl_ex  <- vapply(df, function(z) {
        if (is.factor(z)) paste(utils::head(levels(z), head_levels), collapse = "|") else ""
      }, character(1))
      data.frame(
        column = names(df),
        class  = cls,
        is_character = is_char,
        is_factor    = is_fac,
        n_levels     = lvl_n,
        ex_levels    = lvl_ex,
        row.names = NULL,
        check.names = FALSE
      )
    },

    .stop_with_context = function(msg, Xdf, err = NULL) {
      audit <- private$.audit_df(Xdf)
      bad_chars <- audit$column[audit$is_character]
      cat("=== XGBoost DMatrix construction failed ===\n")
      cat("Reason:", msg, "\n")
      if (!is.null(err)) cat("xgboost says:", conditionMessage(err), "\n")
      cat("Data summary: ", nrow(Xdf), " rows x ", ncol(Xdf), " cols\n", sep = "")
      if (length(bad_chars)) {
        cat("Character columns (must convert to factor or numeric):\n  - ",
            paste(bad_chars, collapse = ", "), "\n", sep = "")
      }
      fac_rows <- audit[audit$is_factor, ]
      if (nrow(fac_rows)) {
        cat("Factor columns and levels (first few):\n")
        utils::capture.output(print(fac_rows[, c("column","n_levels","ex_levels")], row.names = FALSE)) |>
          paste(collapse = "\n") |> cat("\n")
      }
      stop(msg, call. = FALSE)
    },

    .train = function(task) {
      args <- self$params
      verbose <- if (is.null(args$verbose)) getOption("sl3.verbose") else args$verbose
      args$verbose <- as.integer(verbose)

      # outcome
      outcome_type <- self$get_outcome_type(task)
      Y <- outcome_type$format(task$Y)
      if (outcome_type$type == "categorical") Y <- as.numeric(Y) - 1L

      # covariates as raw data.frame (no factor expansion)
      Xdf <- task$get_data(columns = task$nodes$covariates, expand_factors = FALSE)
      stopifnot(is.data.frame(Xdf))

      # convert any characters to factor (xgboost supports factor directly via xgb.DMatrix)
      if (any(vapply(Xdf, is.character, logical(1)))) {
        Xdf[] <- lapply(Xdf, function(z) if (is.character(z)) factor(z) else z)
      }

      print(is.data.frame(Xdf))

      # Build DMatrix with hard check + diagnostics
      dm <- try(xgboost::xgb.DMatrix(data = Xdf, feature_names = colnames(Xdf)), silent = TRUE)
      if (inherits(dm, "try-error") || !inherits(dm, "xgb.DMatrix")) {
        private$.stop_with_context("xgb.DMatrix() did not return an xgb.DMatrix", Xdf, err = attr(dm, "condition"))
      }

      # weights
      if (task$has_node("weights")) {
        try(xgboost::setinfo(dm, "weight", task$weights), silent = TRUE)
      }

      # offset / base_margin
      if (task$has_node("offset")) {
        if (outcome_type$type == "categorical") {
          stop("offsets not yet supported for outcome_type='categorical'")
        }
        family   <- outcome_type$glm_family(return_object = TRUE)
        link_fun <- args$family$linkfun
        offset   <- task$offset_transformed(link_fun)
        try(xgboost::setinfo(dm, "base_margin", offset), silent = TRUE)
      } else {
        link_fun <- NULL
      }

      # reasonable default objective
      if (is.null(args$objective)) {
        if (outcome_type$type == "binomial") {
          args$objective <- "binary:logistic"; args$eval_metric <- "logloss"
        } else if (outcome_type$type == "quasibinomial") {
          args$objective <- "reg:logistic"
        } else if (outcome_type$type == "categorical") {
          args$objective <- "multi:softprob"; args$eval_metric <- "mlogloss"
          args$num_class <- as.integer(length(outcome_type$levels))
        } else {
          # continuous
          if (is.null(args$eval_metric)) args$eval_metric <- "rmse"
        }
      }

      args$data <- dm
      args$watchlist <- list(train = dm)

      fit_object <- call_with_args(xgboost::xgb.train, args, keep_all = TRUE, ignore = "formula")
      fit_object$training_offset   <- task$has_node("offset")
      fit_object$link_fun          <- link_fun
      fit_object$sl3_feature_names <- colnames(Xdf)
      fit_object$sl3_factor_levels <- lapply(Xdf, function(z) if (is.factor(z)) levels(z) else NULL)
      fit_object
    },

    .predict = function(task = NULL) {
      fit <- private$.fit_object

      # raw df
      Xdf <- task$get_data(columns = task$nodes$covariates, expand_factors = FALSE)
      stopifnot(is.data.frame(Xdf))

      # relevel factors to training levels; also coerce stray characters
      tr_lvls_list <- fit$sl3_factor_levels
      for (nm in names(Xdf)) {
        if (is.character(Xdf[[nm]])) Xdf[[nm]] <- factor(Xdf[[nm]])
        tr_lvls <- tr_lvls_list[[nm]]
        if (!is.null(tr_lvls) && is.factor(Xdf[[nm]])) {
          Xdf[[nm]] <- factor(Xdf[[nm]], levels = tr_lvls)
        }
      }

      # reorder columns to training order (critical)
      tr_names <- fit$sl3_feature_names
      ord <- match(tr_names, colnames(Xdf))
      if (anyNA(ord)) {
        missing_cols <- tr_names[is.na(ord)]
        msg <- paste0("Prediction data is missing ", length(missing_cols),
                      " training column(s): ", paste(missing_cols, collapse = ", "))
        private$.stop_with_context(msg, Xdf, err = NULL)
      }
      Xdf <- Xdf[, ord, drop = FALSE]

      # DMatrix (with check)
      dm <- try(xgboost::xgb.DMatrix(data = Xdf, feature_names = colnames(Xdf)), silent = TRUE)
      if (inherits(dm, "try-error") || !inherits(dm, "xgb.DMatrix")) {
        private$.stop_with_context("xgb.DMatrix() failed during predict()", Xdf, err = attr(dm, "condition"))
      }

      # optional offset
      if (fit$training_offset) {
        offset <- task$offset_transformed(fit$link_fun, for_prediction = TRUE)
        try(xgboost::setinfo(dm, "base_margin", offset), silent = TRUE)
      }

      # best_ntreelimit if present
      ntreelimit <- 0
      if (!is.null(fit[["best_ntreelimit"]]) && !("gblinear" %in% fit[["params"]][["booster"]])) {
        ntreelimit <- fit[["best_ntreelimit"]]
      }

      if (nrow(Xdf) == 0) return(numeric(0))

      preds <- stats::predict(fit, newdata = dm, ntreelimit = ntreelimit, reshape = TRUE)
      if (private$.training_outcome_type$type == "categorical") preds <- pack_predictions(preds)
      preds
    },

    .required_packages = c("xgboost")
  )
)
