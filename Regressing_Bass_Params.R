library(tidyverse)
library(MASS)
library("geoR")

# Read params and census variables file
params_variables_census <- read_csv('params_variables_old3.csv') # Much better performance than new zones
weather_data <- read_csv('weather_vars_zones.csv')
gas_station_counts <- read_csv('gas_stations_count_old_zones.csv')
province <- read_csv('province.csv')
cma_summarized <- read_csv('cma_summarized.csv')
charging_stns_counts <- read_csv('charging_stations_count_zones_old.csv')
oldzone_access <- read_csv('oldzone_access.csv')
land_use <- read_delim('LU_Areas_on_NewZones.csv', delim = '\t')
incentives_regression_input <- read_csv('incentives_regression_input_grouped_25_07.csv')
charging_stns_new_input <- read_csv('charging_stations_per_area.csv')

params_variables_census <- inner_join(params_variables_census, weather_data) 
params_variables_census <- inner_join(params_variables_census, province)
params_variables_census <- inner_join(params_variables_census, gas_station_counts)
params_variables_census <- inner_join(params_variables_census, cma_summarized)
params_variables_census <- inner_join(params_variables_census, charging_stns_counts)
params_variables_census <- inner_join(params_variables_census, oldzone_access)
params_variables_census <- inner_join(params_variables_census, land_use)
params_variables_census <- inner_join(params_variables_census, incentives_regression_input)
params_variables_census <- inner_join(params_variables_census, charging_stns_new_input)



# ## File for storing results
# txt_conn <- file("bass_model_explanatory_vars_19_08.txt", open = "wt")


# Initial log-normal regression
params_variables_census <- params_variables_census |> 
  mutate(
    logp = log(p),
    logq = log(q), 
    prop_engineers = COL38 / COL35, 
    prop_arts = COL41 / COL35,
    prop_agric = COL45 / COL35,
    PR_factor = as.factor(Province),
    pr_BC = if_else(Province == 'BritishColumbia', 1, 0), 
    pr_ON = if_else(Province == 'Ontario', 1, 0),
    pr_QC = if_else(Province == 'Quebec', 1, 0), 
    inc_BC = if_else(Province == 'BritishColumbia', COL5, 0), 
    inc_ON = if_else(Province == 'Ontario', COL5, 0), 
    inc_QC = if_else(Province == 'Quebec', COL5, 0), 
    gas_stations_sufficient = if_else(gas_stations_count  > 20, 1, 0), 
    gas_stations_per_capita = gas_stations_count / COL31, 
    gas_stations_per_sqkm = gas_stations_count / COL32, 
    charging_stations_per_capita = charging_stations_count / COL31, 
    charging_stations_per_sqkm = charging_stations_count / COL31, 
    median_zonal_age = COL33,
    residential_commercial_ratio = Residentia / Commercial,
    residential_open_ratio = Residentia / OpenArea_O,
    commercial_open_ratio = Commercial / OpenArea_O
  )

# write.csv2(params_variables_census, "all_vars.csv")

bcp <- boxcox(p ~ MIN_TEMPERATURE + prop_engineers + pr_BC + prop_arts  + pct_condominium
              , data = params_variables_census)
(lambda_p <- bcp$x[which.max(bcp$y)])

bcq <-boxcox(q ~  COL33 + MIN_TEMPERATURE + pr_BC + pr_QC, data = params_variables_census)
(lambda_q <- bcq$x[which.max(bcq$y)])

params_variables_census <- params_variables_census |> 
  mutate(
    p_tr = (p ^ lambda_p - 1) / lambda_p,
    q_tr = (q ^ lambda_q - 1) / lambda_q,
    weights_p = 1 / p_se,
    weights_q = 1 / q_se, 
    log_weights_p = log(weights_p)
  )

# Initial regression 
weighted_reg_p <- lm(p_tr ~ PR_factor + prop_arts + prop_agric + pct_condominium 
                     + median_zonal_age 
                     # + Government 
                     + weighted_avg_subsidy
                     # + Charging_stations
                     # + Charging_stations_density
                     , data = params_variables_census, weights = weights_p)
summary(weighted_reg_p)

# Write summary
p_summary <- summary(weighted_reg_p)
capture_output <- capture.output(print(p_summary))
cat(paste(capture_output, collapse = "\n"), "\n\n", file = txt_conn)

# Save coeffs to csv
coef_mat_p <- p_summary$coefficients
write.csv(coef_mat_p, "p_regression_weighted_results_19_08.csv")


# Log lin version q
weighted_reg_q <- lm(q_tr ~ pr_ON + pr_BC + pr_QC + commute_long_proportion 
                     # + weighted_avg_subsidy
                     # + Charging_stations_density
                     + Charging_stations
                     , weights = weights_q, data = params_variables_census)
summary(weighted_reg_q)


# Write summary
q_summary <- summary(weighted_reg_q)
capture_output <- capture.output(print(q_summary))
cat(paste(capture_output, collapse = "\n"), "\n\n", file = txt_conn)

# Save coeffs to csv
coef_mat_q <- q_summary$coefficients
write.csv(coef_mat_q, "q_regression_weighted_results_19_08.csv")


# # Close the text file connection
# close(txt_conn)
