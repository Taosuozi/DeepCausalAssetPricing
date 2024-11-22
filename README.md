# Topic: Asset Pricing Analysis Based on Deep Learning Causal Inference

## Overview

### 1. Stock Market Asset Inference Principles Based on Mispricing

This section refers to the following paper:
 Bartram S. M., Grinblatt M. *Agnostic fundamental analysis works*. Journal of Financial Economics, 2018, 128(1): 125-147.

- Construct mispricing variables using fundamental indicators of stocks.
- Present the formula for mispricing (as described in the paper, note that the PDF contains errors in the formula).
- Explain the rationale for choosing this approach (as outlined in the literature).

**Table 1: Variable Description** (from the PDF)

**Table 2: Descriptive Statistics of All Variables**

------

### 2. Mispricing Variable Construction Using the Adaboost Algorithm

**Principle:**

- Use the Adaboost algorithm and fundamental factors of all stocks on the same date to group and predict the mispricing variable for each stock.
- After prediction, group by date. For each date, conduct stock selection and investment:
  - Rank stocks by their mispricing values in descending order.
  - Compare two investment strategies:
    - **Q5**: Buy the top 20% of stocks and sell them after *x* months.
    - **Q1**: Buy the bottom 20% of stocks and sell them after *x* months.
- Compare the returns of Q5 and Q1.

**Results:**

- The average returns of Q5 are generally higher than those of Q1.
- The gap between Q5 and Q1 widens over time, demonstrating the relevance of mispricing variables in asset pricing and their delayed effectiveness.

**Table 3: Investment Portfolio Performance Constructed Using Mispricing M**

------

### 3. Causal Inference Analysis of the Mispricing Variable (M)

This section primarily refers to the following sections in the PDF:

- **1. Basic Model of Causal Diagrams**
- **2. Variational Autoencoder (VAE) Deep Learning Model**
- **4.3 Causal Inference Between Mispricing and Market Capitalization Variables: ITE (Individual Treatment Effect)**

**Table 4: Individual Treatment Effect (ITE) of Mispricing Variable M and Market Capitalization Variable**

| **Variable**                          | **ITE**                | **T-statistic** | **P-value** |
| ------------------------------------- | ---------------------- | --------------- | ----------- |
| Mispricing Variable M                 | 735671877861023        | 33.9656         | 0.0000      |
| Market Capitalization (Control Group) | -0.0005595158544303851 | -27.8666        | 0.0000      |

**Conclusion:**

- The ITE of the mispricing variable is significant, indicating a positive causal effect of mispricing on asset returns.
- The ITE of the market capitalization variable is smaller, suggesting a weaker causal effect of market capitalization on asset returns.