library(dplyr)
library(readr)
library(ggplot2)

# Folders to save plots and vcov matrices
output_folder <- "market_size_90_perc_correct_hev"
vcov_folder   <- "vcovs_16_05_05c"
if (!dir.exists(output_folder)) dir.create(output_folder)
if (!dir.exists(vcov_folder))   dir.create(vcov_folder)

# File names
txt_file_name <- "results_txt_m_p_q_provided_90_perc_hev.txt"
csv_name      <- "results_csv_m_p_q_provided_90_perc_hev.csv"

# Read input data
bass_input   <- read_csv("bass_input_zones_hev.csv")
market_sizes <- read_csv("market_size_90_perc_correct2.csv")      # must include ZoneID & Total
start_params <- read_csv("best_parameter_market_size_90_perc_correct_hev.csv")

# Bass model function
c_t <- function(x, p, q, m) {
  num   <- 1 - (p + q)
  denom <- 1 + (p + q)
  m * (
    ((1 - ((num/denom)^(x + 1))/2) / (1 + (q/p) * ((num/denom)^(x + 1))/2)) -
      ((1 - ((num/denom)^(x - 1))/2) / (1 + (q/p) * ((num/denom)^(x - 1))/2))
  )
}

zone_list <- unique(bass_input$ZoneID)

# Prepare text output
txt_conn <- file(txt_file_name, open = "wt")

# Initialize results data frame with R2 & RMSE
results_df <- tibble(
  ZoneID = character(),
  p      = double(), p_se = double(), p_t = double(),
  q      = double(), q_se = double(), q_t = double(),
  m      = double(),
  R2     = double(),
  RMSE   = double()
)

for (zone in zone_list) {
  input_data <- bass_input %>%
    filter(ZoneID == zone, months_passed_01_2021 > 0)
  
  # market size and starting values
  m_value <- market_sizes %>% filter(ZoneID == zone) %>% pull(Total) %>% first()
  p_start <- start_params %>% filter(ZoneID == zone)  %>% pull(p)
  q_start <- start_params %>% filter(ZoneID == zone)  %>% pull(q)
  
  cat("ZoneID:", zone, "\n", file = txt_conn)
  
  if (is.na(m_value)) {
    cat("  Market size missing\n\n", file = txt_conn)
    results_df <- results_df %>% add_row(
      ZoneID = zone, p=NA, p_se=NA, p_t=NA,
      q=NA, q_se=NA, q_t=NA,
      m=NA, R2=NA, RMSE=NA
    )
    next
  }
  
  if (nrow(input_data) == 0) {
    cat("  No data points\n\n", file = txt_conn)
    results_df <- results_df %>% add_row(
      ZoneID = zone, p=NA, p_se=NA, p_t=NA,
      q=NA, q_se=NA, q_t=NA,
      m=m_value, R2=NA, RMSE=NA
    )
    next
  }
  
  input_data <- input_data %>% mutate(m_fixed = m_value)
  
  # Fit model
  fit <- tryCatch(
    nls(`2021-2024` ~ c_t(months_passed_01_2021, p, q, m_fixed),
        data      = input_data,
        start     = list(p = p_start, q = q_start),
        algorithm = "port",
        control   = nls.control(maxiter = 1e7, minFactor = 1e-4)),
    error = function(e) NULL
  )
  
  if (is.null(fit)) {
    cat("  Fit failed\n\n", file = txt_conn)
    results_df <- results_df %>% add_row(
      ZoneID = zone, p=NA, p_se=NA, p_t=NA,
      q=NA, q_se=NA, q_t=NA,
      m=m_value, R2=NA, RMSE=NA
    )
    next
  }
  
  # 1) Summary & text file
  fit_sum <- summary(fit)
  capture.output(fit_sum, file = txt_conn, append = TRUE)
  
  # 2) Goodness-of-fit
  observed  <- input_data$`2021-2024`
  predicted <- predict(fit, newdata = input_data)
  sse       <- sum((observed - predicted)^2)
  sst       <- sum((observed - mean(observed))^2)
  r2        <- 1 - sse / sst
  rmse      <- sqrt(mean((observed - predicted)^2))
  
  cat(sprintf("\n  R-squared: %.4f\n  RMSE:      %.4f\n\n", r2, rmse),
      file = txt_conn)
  
  # 3) Append to results_df
  coefs <- fit_sum$coefficients
  results_df <- results_df %>% add_row(
    ZoneID = zone,
    p      = coefs["p","Estimate"],
    p_se   = coefs["p","Std. Error"],
    p_t    = coefs["p","t value"],
    q      = coefs["q","Estimate"],
    q_se   = coefs["q","Std. Error"],
    q_t    = coefs["q","t value"],
    m      = m_value,
    R2     = r2,
    RMSE   = rmse
  )
  
  # 4) Save variance–covariance matrix
  vcov_mat <- vcov(fit)
  write.csv(vcov_mat,
            file = file.path(vcov_folder, paste0(zone, ".csv")),
            row.names = TRUE)
  
  # 5) Plot observed vs predicted
  plot_df <- input_data %>% mutate(predicted = predicted)
  p_plot <- ggplot(plot_df, aes(months_passed_01_2021)) +
    geom_line(aes(y = `2021-2024`, color = "Observed")) +
    geom_line(aes(y = predicted,    color = "Predicted")) +
    labs(title = paste("ZoneID:", zone),
         x     = "Months since Jan 2021",
         y     = "Adoption") +
    theme_minimal() +
    scale_color_manual("", values = c("Observed"="black","Predicted"="red"))
  
  ggsave(
    filename = file.path(output_folder, paste0("zone_", zone, ".png")),
    plot     = p_plot, width = 6, height = 4
  )
}

close(txt_conn)

# 6) Write overall summary CSV
write_csv(results_df, csv_name)