# Exploring-Racial-Bias-in-Credit-Approvals | Jan’25–Apr’25**

## Project Overview

This project investigates the presence and extent of racial bias in home loan approvals, focusing specifically on Black (African American) applicants. Using the Home Mortgage Disclosure Act (HMDA) dataset, the analysis employs both Maximum Likelihood Estimation (MLE) and Bayesian methods to estimate the probability of loan denial and to quantify the impact of race on credit decisions.

## Objective

- **Estimate** the probability of loan denial for mortgage applicants.
- **Analyze** the effect of race—particularly for Black families—on loan approval outcomes.
- **Compare** traditional (MLE) and Bayesian approaches to inference in this context.

## Methodology

- **Data Source:**  
  - Home Mortgage Disclosure Act (HMDA) dataset, a comprehensive, publicly available resource for analyzing mortgage lending practices and potential discrimination[1].
- **Statistical Models:**  
  - **Probit regression** estimated via Maximum Likelihood Estimation (MLE) to predict loan denial.
  - **Bayesian probit regression** using Markov Chain Monte Carlo (MCMC) with 20,000 iterations and 5,000 burn-in, applying a normal prior to estimate the posterior distribution of 20 model coefficients.
- **Key Variable:**  
  - The `afam` (African American) covariate is used to isolate and measure the effect of being Black on loan denial probability.

## Key Results

- The coefficient for the `afam` variable is **positive and statistically significant** in both MLE (0.3957, SE = 0.0986) and Bayesian (mean = 0.3972, SD = 0.1017) models, confirming robust evidence of racial bias against Black applicants.
- Both estimation approaches yield closely aligned results, with the **Average Causal Effect (ACE) = 0.158**, indicating that being Black increases the probability of loan denial by **15.8%**, holding other factors constant.
- These findings are consistent with recent literature documenting persistent racial disparities in mortgage lending, even after accounting for income and other relevant characteristics[2][1][3].

## Policy & Social Implications

- The analysis reinforces the need for ongoing regulatory scrutiny and policy interventions to address racial disparities in credit markets.
- Lenders and policymakers must ensure that automated and human-driven loan approval processes are free from discriminatory practices, as even advanced statistical models and machine learning systems can perpetuate or amplify existing biases[2][4][3].

## Technical Skills & Tools

- **Statistical Modeling:** Probit regression, Bayesian inference, MCMC.
- **Programming:** R, Python (PyMC3/Stan), data preprocessing and visualization.
- **Data Analysis:** Pandas, NumPy, interpretation of statistical outputs.

## References

- HMDA dataset documentation and related research on mortgage discrimination[1][3].
- Recent studies on racial bias in algorithmic lending and credit approval systems[2][4].

## Acknowledgments

Project completed as part of ECO545 under the supervision of **Prof. Arshad Rahman** at the Indian Institute of Technology Kanpur.


**Keywords:** Racial Bias, Credit Approvals, Bayesian Probit, MLE, HMDA, Mortgage Lending, Discrimination Analysis
