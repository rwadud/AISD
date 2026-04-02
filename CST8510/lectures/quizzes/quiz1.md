# Quiz

## Question 1

You are working on a predictive modeling problem where the goal is to classify whether an email is spam or not spam. The dataset contains 10,000 emails, each with the following features:

- Text content of the email (natural language text).
- Sender's email address (categorical).
- Time the email was sent (continuous).
- Number of recipients (continuous).

The dataset is highly imbalanced, with only 5% of the emails being spam. You want to build a model that not only achieves high accuracy but also has a good recall for the spam class (i.e., correctly identifies as many spam emails as possible).

Which of the following machine learning algorithms is most suitable for this problem?

- A) Random Forest
- B) Gradient Boosting (XGBoost, LightGBM)
- C) Logistic Regression
- D) Support Vector Machine (SVM) with a linear kernel

## Question 2

A team is developing a machine translation system to translate technical documentation from English to German. They have these specific requirements:

- Must maintain technical accuracy
- Word order is important for readability
- Must preserve the exact meaning of technical terms

Given these sample translations:

**Original English:** "The database server processes queries in parallel using multiple CPU cores."

**Translation 1:** "Der Datenbankserver verarbeitet Abfragen parallel unter Verwendung mehrerer CPU-Kerne." (Perfect word-by-word translation)

**Translation 2:** "Unter Verwendung mehrerer CPU-Kerne verarbeitet der Server Datenbankabfragen parallel." (Same meaning but different word order)

**Translation 3:** "Der Server nutzt verschiedene Prozessoren für parallele Datenbankabfragen." (Slightly different terminology but same general meaning)

Which evaluation metric would be MOST appropriate for this specific use case?

- A) BLEU Score
- B) Simple word error rate calculation
- C) BERTScore
- D) ROUGE Score

## Question 3

You are building a machine learning model to predict whether a customer will purchase a product. The dataset contains a categorical feature called Color, which has the following unique values: Red, Blue, Green, Yellow.

Which of the following is the best way to represent this feature for use in a machine learning model?

- A) Assign numbers to each color randomly: Red = 1, Blue = 2, Green = 3, Yellow = 4
- B) Assign numbers to each color based on alphabetical order: Blue = 1, Green = 2, Red = 3, Yellow = 4
- C) Create separate binary columns for each color: Red = [1, 0, 0, 0], Blue = [0, 1, 0, 0], Green = [0, 0, 1, 0], Yellow = [0, 0, 0, 1]
- D) Use the original text values directly in the model: Red, Blue, Green, Yellow

## Question 4

Below is a dataset with 8 rows and 6 columns, simulating a clinical study. The dataset contains both numerical and categorical variables, and the missing values are denoted by NaN. The dataset is an example of what type of missing values?

| Patient ID | Age | Gender | Treatment Group | Baseline BP (mmHg) | Follow-Up BP (mmHg) | Medication Adherence |
|---|---|---|---|---|---|---|
| 1 | 45 | Male | Control | 140 | 138 | High |
| 2 | 52 | Female | Treatment | 145 | NaN | Medium |
| 3 | 60 | Male | Treatment | 150 | 142 | NaN |
| 4 | 38 | Female | Control | 135 | 132 | NaN |
| 5 | 47 | Male | Treatment | 148 | NaN | High |
| 6 | 55 | Female | Control | 142 | 139 | NaN |
| 7 | 50 | Male | Treatment | 155 | 148 | Medium |
| 8 | 43 | Female | Control | 137 | NaN | Low |

- A) Missing Completely At Random
- B) Missing At Random
- C) Missing Not At Random
- D) None of the Above

## Question 5

You are working on a machine learning model to predict customer churn for a telecom company. After training the model, you use SHAP values to explain the model's predictions. One of the customers, Alice, has a high predicted probability of churn. The SHAP value for the feature MonthlyCharges is +0.15 for Alice's prediction.

What does this SHAP value indicate?

- A) Alice's MonthlyCharges value is higher than the average MonthlyCharges in the dataset, and this decreases her predicted probability of churn.
- B) Alice's MonthlyCharges value is higher than the average MonthlyCharges in the dataset, and this increases her predicted probability of churn.
- C) Alice's MonthlyCharges value is lower than the average MonthlyCharges in the dataset, and this increases her predicted probability of churn.
- D) Alice's MonthlyCharges value is lower than the average MonthlyCharges in the dataset, and this decreases her predicted probability of churn.

## Question 6

An LLM gives this response to a user's question about "What did Einstein contribute to physics?":

> "Einstein revolutionized physics with his Theory of Relativity in 1905. He discovered that E=mc², showed that light behaves as both particles and waves, and proved that time passes differently based on gravity and speed. He also worked closely with Isaac Newton and collaborated on many experiments about gravity at Princeton University."

Which evaluation metric would BEST identify the key issue in this response?

- A) Faithfulness
- B) Answer Relevancy
- C) Prompt Alignment
- D) Contextual Grounding

## Question 7

For the LLM-As-A-Judge method, what is the type of Bias where an LLM prefers the text generated by an LLM from the same family?

- A) Attention Bias
- B) Nepotism Bias
- C) Beauty Bias
- D) Authority Bias

## Question 8

Say True or False: In the case of Observability Data from Open Telemetry, Span is a smaller unit of data compared to a Trace.

- True
- False

## Question 9

You are building a machine learning model to predict stock prices for a company. During feature engineering, you create a new feature called Avg_Price_Last_7_Days, which calculates the average stock price over the past 7 days for each day in your dataset. You split your data into training and testing sets using a time-based split, where the training set contains data from January to September, and the testing set contains data from October to December. You compute the Avg_Price_Last_7_Days feature separately for the training and testing sets after split.

True or False: Is this the correct way of computing features?

- True
- False
