# Lecture 12 Exam Questions and Answers

This sheet focuses on likely exam style questions from the lecture review, especially questions that require little or no calculation.

## 1. Conceptual Questions

### 1. Why is accuracy misleading in imbalanced classification?

**Answer**: When one class is much larger than the other, a classifier can get high accuracy by predicting only the majority class. For example, if almost all samples are negative, predicting everything as negative can still produce very high accuracy while completely missing the minority class. In these problems, the minority class is often the class that matters most.

### 2. Why are precision and recall often more important than accuracy for imbalanced data?

**Answer**: Precision tells us how trustworthy positive predictions are, and recall tells us how many actual positive cases we successfully found. In imbalanced problems, these measures reveal whether the model can detect the rare but important class, while accuracy may hide failure.

### 3. When is precision more important than recall?

**Answer**: Precision is more important when false positives are costly. A good example is when you want only highly relevant search results, or when a false alarm creates expensive follow up work.

### 4. When is recall more important than precision?

**Answer**: Recall is more important when missing a true positive is costly. Disease detection is a common example. Missing a sick patient is usually worse than checking some healthy people more carefully.

### 5. What does an ROC curve show?

**Answer**: An ROC curve shows the relationship between the true positive rate and the false positive rate as the decision threshold changes.

### 6. What does AUC mean?

**Answer**: AUC is the area under the ROC curve. In general, a larger AUC means better overall ranking performance than a smaller AUC.

### 7. Why might the model with the best overall AUC still not be best for every sample?

**Answer**: Different models can perform better in different regions of the data. One model may work better for one cluster of samples, while another model may work better for another cluster.

### 8. What is the simplest way to handle class imbalance?

**Answer**: Resampling. We can undersample the majority class or oversample the minority class.

### 9. What is the difference between random oversampling and SMOTE?

**Answer**: Random oversampling duplicates existing minority samples. SMOTE creates new synthetic minority samples by interpolating between nearby minority samples.

### 10. Why does SMOTE work naturally for numeric data?

**Answer**: SMOTE assumes that samples live in a vector space where points between two nearby samples still make sense. This assumption fits ordinary numeric features.

### 11. Why is naive SMOTE usually not appropriate for raw images?

**Answer**: Direct interpolation in pixel space often creates unrealistic samples. Raw images are highly structured, so interpolated pixel values may not represent meaningful images.

### 12. What is the difference between an outlier and noise?

**Answer**: An outlier is unusual but can still be valid. Noise is invalid or corrupted data. A salary of 900K might be unusual but valid. A birth date in the wrong format is noise.

### 13. Why does context matter in anomaly detection?

**Answer**: A sample that looks unusual in one dataset may be normal in another. A very tall person may be unusual in a general population dataset, but normal in a basketball player dataset.

### 14. How can an autoencoder be used for compression?

**Answer**: The encoder maps a high dimensional input into a smaller hidden representation. We can store that shorter representation and decode it later.

### 15. How can an autoencoder be used for denoising?

**Answer**: We give the noisy sample as input and the clean sample as the target output. The model learns to remove noise during reconstruction.

### 16. How can an autoencoder be used for anomaly detection?

**Answer**: If the model is trained on normal data, it usually reconstructs normal samples well. Abnormal samples produce larger reconstruction error, so the error can be used as an anomaly score.

### 17. What does one class SVM learn?

**Answer**: It learns a boundary around the normal data. Samples outside that learned region are treated as outliers.

### 18. What is the origin trick in one class SVM?

**Answer**: In feature space, the origin is treated as the opposing reference. The model learns a hyperplane that separates the normal data from the origin with a large margin while allowing some slack for outliers.

### 19. In one class SVM, what does it mean if f of x is greater than or equal to zero?

**Answer**: The sample is treated as normal.

### 20. In one class SVM, what does the hyperparameter nu control?

**Answer**: It controls how tight or loose the learned normal region is, and the expected fraction of outliers.

### 21. What is classifier fusion or ensemble learning?

**Answer**: It is the idea of building several classifiers and combining their predictions to get a final decision.

### 22. Why is classifier diversity important in ensemble learning?

**Answer**: If all classifiers behave the same way, combining them adds little value. Diversity helps the ensemble correct individual errors.

### 23. Why can weighted majority voting be better than simple majority voting?

**Answer**: Weighted voting gives more influence to stronger classifiers. Simple majority voting treats all classifiers as equally reliable, even when they clearly are not.

### 24. What are the three main stages of an ensemble learning pipeline?

**Answer**: Classifier generation, classifier selection, and classifier combination.

### 25. What is forward selection in classifier selection?

**Answer**: Start with the best classifier, then add one classifier at a time, keeping a new classifier only if it improves overall ensemble performance.

### 26. What is backward elimination in classifier selection?

**Answer**: Start with all classifiers, then remove low performing classifiers one by one, keeping a removal only if performance improves.

### 27. Why is selecting a subset from a very large classifier pool hard?

**Answer**: Because the number of possible subsets becomes enormous. This makes the problem combinatorial.

### 28. Which types of base classifiers often work well in ensembles?

**Answer**: Unstable classifiers such as decision trees and neural networks. They are sensitive to changes in data or hyperparameters, so combining them can reduce variance.

### 29. What is the relationship between overfitting, underfitting, bias, and variance?

**Answer**: Overfitting corresponds to low bias and high variance. Underfitting corresponds to high bias and low variance.

### 30. What can cause overfitting?

**Answer**: Excessive model complexity, small datasets, and training that allows the model to memorize details.

### 31. What can cause underfitting?

**Answer**: A model that is too simple, insufficient training, poor data quality, or not enough useful signal in the features.

### 32. What is bagging?

**Answer**: Bagging stands for bootstrap aggregating. We create multiple bootstrap samples, train one classifier on each sample, and then combine their predictions.

### 33. What is a bootstrap sample?

**Answer**: It is a sample drawn with replacement from the training data. It can contain duplicates, and some original samples may be missing.

### 34. Why does bagging create diversity?

**Answer**: Different bootstrap samples lead to different training sets, which can produce different classifiers even when the same learning algorithm is used.

### 35. What is a weakness of ensemble methods from an explanation point of view?

**Answer**: The final decision is harder to explain clearly. Saying that many classifiers voted is less transparent than explaining one simple classifier.

### 36. What is the main idea of boosting?

**Answer**: Boosting focuses more on difficult samples that earlier classifiers got wrong, instead of treating every sample equally in every round.

### 37. How does boosting update sample weights?

**Answer**: It increases the weights of misclassified samples and decreases the weights of correctly classified samples.

### 38. Why is boosting harder to parallelize than bagging?

**Answer**: Because each boosting round depends on the updated weights from the previous round. Bagging rounds are independent.

### 39. What does AdaBoost add beyond basic boosting?

**Answer**: AdaBoost assigns weights both to samples and to classifiers. Stronger classifiers get more influence in the final decision.

### 40. What happens in AdaBoost if a classifier performs worse than random guessing?

**Answer**: Its error is too high to trust for reweighting. The weights are reset and resampling is repeated.

### 41. What makes random forests different from ordinary bagging?

**Answer**: Random forests use decision trees as base learners and also randomize the features by selecting only a subset of features at each split.

### 42. What is association rule mining trying to find?

**Answer**: Rules that describe which items tend to occur together in transactions.

### 43. What is an itemset?

**Answer**: A collection of one or more items.

### 44. What is support count?

**Answer**: The number of transactions that contain an itemset.

### 45. What is support?

**Answer**: Support is support count divided by the total number of transactions.

### 46. What is confidence in an association rule?

**Answer**: Confidence measures how often Y appears among transactions that contain X in the rule X implies Y.

### 47. What is a frequent itemset?

**Answer**: An itemset whose support is at least the minimum support threshold.

### 48. What are minsup and minconf?

**Answer**: They are user chosen hyperparameters. Minsup is the minimum support threshold, and minconf is the minimum confidence threshold.

### 49. What is the Apriori principle?

**Answer**: If an itemset is frequent, then all of its subsets must also be frequent. Equivalently, if an itemset is infrequent, then all of its supersets must also be infrequent.

### 50. Why is the Apriori principle useful?

**Answer**: It reduces the search space. Once we know a set is infrequent, we can prune all larger sets that contain it.

### 51. What is one limitation of a minimum support based association rule framework?

**Answer**: Two items may always occur together but still be missed if their total support is below the minimum support threshold.

### 52. What is the difference between a hyperparameter and a parameter?

**Answer**: A hyperparameter is chosen before training. A parameter is learned from the data during training.

### 53. Give one example of a hyperparameter and one example of a learned parameter.

**Answer**: In K means, the number of clusters is a hyperparameter. The final centroids are learned parameters.

### 54. What are some common ways to tune hyperparameters?

**Answer**: Manual search, grid search, random search, and choosing values based on prior literature or common best practices.

## 2. Low Math Practice Questions

### 55. A dataset has 1,000 samples, 10 positive and 990 negative. A classifier predicts every sample as negative. Why is the classifier still poor even though accuracy is 99 percent?

**Answer**: It misses all 10 positive cases. In an imbalanced problem, that means it completely fails to detect the minority class.

### 56. If true positives equal 8, false positives equal 4, and false negatives equal 2, which is larger, precision or recall?

**Answer**: Precision is $8 / 12$, while recall is $8 / 10$. Recall is larger.

### 57. A one class SVM gives these decision values: sample A has f of x equals 0.8, sample B has f of x equals 0, sample C has f of x equals negative 0.2. Which sample is an outlier?

**Answer**: Sample C is the outlier, because f of x is less than zero.

### 58. Three classifiers vote on one sample. Two vote negative with weights 0.3 and 0.2. One votes positive with weight 0.9. What is the final class under weighted majority voting?

**Answer**: Positive. The positive side has total weight 0.9, while the negative side has total weight 0.5.

### 59. A bootstrap sample is drawn from records A, B, C, and D. The sample becomes {A, B, B, D}. What two bagging properties does this show?

**Answer**: It shows duplication, because B appears twice, and omission, because C is missing.

### 60. In boosting, if one sample is repeatedly misclassified, what happens to its weight?

**Answer**: Its weight increases, so it is more likely to appear in later rounds.

### 61. In random forest training, why does choosing only a subset of features at each split help?

**Answer**: It makes different trees less similar to each other, which increases diversity and reduces variance.

### 62. If an itemset {bread, milk} is infrequent, what can Apriori say about {bread, milk, butter}?

**Answer**: It must also be infrequent, so it can be pruned.

### 63. Suppose there are 5 transactions, and {bread, milk} appears in 3 of them. What is the support?

**Answer**: Support is $3 / 5$, or 0.6.

### 64. Suppose bread appears in 4 transactions, and bread together with milk appears in 3 transactions. What is the confidence of bread implies milk?

**Answer**: Confidence is $3 / 4$, or 0.75.

### 65. If an autoencoder trained on normal data gives reconstruction error 0.02 for one sample and 1.40 for another, which sample is more likely to be anomalous?

**Answer**: The sample with reconstruction error 1.40 is more likely to be anomalous.

## 3. How To Recognize Small Worked Application Questions

These questions are usually not design questions. They usually give you a tiny dataset, a few numbers, or a short scenario, then ask you to apply one definition or one rule carefully.

### Association Rule Mining

**Question type**: tiny transaction set application question.

**What it usually looks like**:

* You are given 4 or 5 transactions
* You are asked for support count, support, confidence, or whether Apriori can prune something

**What the lecturer is testing**:

* Can you count transactions carefully
* Can you distinguish support count from support
* Can you distinguish support from confidence
* Can you apply the Apriori principle

**How to answer**:

1. Identify the itemset or rule.
2. Count how many transactions contain the itemset.
3. If support is asked, divide by total number of transactions.
4. If confidence is asked, divide support of `X union Y` by support of `X`.
5. If Apriori pruning is asked, use the rule that an infrequent subset makes every superset infrequent.

**Likely exam questions**:

* "Find the support count of `{bread, milk}`."
* "Find the support of `{bread, milk}`."
* "Find the confidence of `bread -> milk`."
* "If `{bread, milk}` is infrequent, what can you say about `{bread, milk, butter}`?"
* "If `minsup` is 0.7, is `{bread, milk}` frequent?"

### Confusion Matrix Metrics

**Question type**: small metric application question.

**What it usually looks like**:

* You are given `TP`, `FP`, `FN`, and sometimes `TN`
* You are asked to compute or compare precision, recall, specificity, false positive rate, or false negative rate

**What the lecturer is testing**:

* Whether you know which counts belong in each formula
* Whether you understand what each metric means

**How to answer**:

1. Write the requested definition first.
2. Substitute the small numbers.
3. If the math is simple, compute it.
4. State the interpretation in one sentence.

**Likely exam questions**:

* "Which is larger, precision or recall?"
* "What is the false positive rate?"
* "Why is accuracy misleading here?"

### ROC and Threshold Questions

**Question type**: threshold effect or one-point ROC question.

**What it usually looks like**:

* You are given a tiny confusion summary at one threshold
* Or you are asked what happens when the threshold is lowered

**What the lecturer is testing**:

* Whether you know that ROC uses true positive rate and false positive rate
* Whether you understand the threshold tradeoff

**How to answer**:

1. Compute true positive rate if needed.
2. Compute false positive rate if needed.
3. State the ROC point or the threshold effect.
4. If the threshold is lowered, say that true positives often increase and false positives often increase too.

**Likely exam questions**:

* "What ROC point do these counts produce?"
* "What happens if we lower the threshold?"
* "Why is the model with the best AUC not always best for every sample?"

### Outlier Versus Noise Questions

**Question type**: classify the scenario.

**What it usually looks like**:

* You are given one strange record
* You are asked whether it is an outlier or noise

**What the lecturer is testing**:

* Whether you know that outliers can be valid
* Whether you know that noise is invalid or corrupted
* Whether you remember that context matters

**How to answer**:

1. Ask whether the sample is unusual but still possible.
2. If yes, call it an outlier.
3. If the value is invalid, corrupted, or wrongly formatted, call it noise.
4. Mention context if it changes the answer.

**Likely exam questions**:

* "Is a 900K salary an outlier or noise?"
* "Is a wrong birth date format an outlier or noise?"
* "Why might a tall person be normal in one dataset but anomalous in another?"

### One Class SVM Questions

**Question type**: decision interpretation question.

**What it usually looks like**:

* You are given `f(x)` values
* Or you are asked what `nu` does

**What the lecturer is testing**:

* Whether you know the sign rule
* Whether you understand that the method learns a normal-data boundary

**How to answer**:

1. Use the lecture rule directly.
2. If `f(x) >= 0`, call the sample normal.
3. If `f(x) < 0`, call the sample an outlier.
4. For `nu`, explain that it controls how tight or loose the normal region is.

**Likely exam questions**:

* "Which sample is an outlier if `f(x)` is negative?"
* "What does `nu` control?"
* "Why is one class SVM useful when anomaly labels are unavailable?"

### Ensemble Voting Questions

**Question type**: tiny vote aggregation question.

**What it usually looks like**:

* You are given 3 or 4 classifier outputs
* Sometimes they have weights
* You are asked for the final decision

**What the lecturer is testing**:

* Whether you can distinguish majority voting from weighted majority voting
* Whether you understand why stronger classifiers should matter more

**How to answer**:

1. If it is simple majority, count votes.
2. If it is weighted voting, sum weights on each side.
3. Pick the larger total.
4. State why the result makes sense.

**Likely exam questions**:

* "What is the final class under majority voting?"
* "What is the final class under weighted majority voting?"
* "Why can weighted voting be better than simple majority voting?"

### Bagging and Boosting Mini Questions

**Question type**: property identification or next-step reasoning question.

**What it usually looks like**:

* You are shown a bootstrap sample
* Or you are told a sample was misclassified repeatedly
* Or you are asked which method is easier to parallelize

**What the lecturer is testing**:

* Whether you know what bootstrap sampling does
* Whether you know how boosting changes sample weights
* Whether you know the bagging versus boosting tradeoff

**How to answer**:

1. For bagging, mention sampling with replacement, duplicates, and missing original samples.
2. For boosting, say misclassified samples get higher weight.
3. For parallelization, prefer bagging because rounds are independent.

**Likely exam questions**:

* "What properties does this bootstrap sample show?"
* "What happens to the weight of a repeatedly misclassified sample?"
* "Which is easier to parallelize, bagging or boosting?"

### Hyperparameter Versus Parameter Questions

**Question type**: label-the-role question.

**What it usually looks like**:

* You are given one setting and one learned quantity
* You are asked which is the hyperparameter and which is the parameter

**What the lecturer is testing**:

* Whether you know what is chosen before training and what is learned from data

**How to answer**:

1. Ask whether the value is chosen before training or learned during training.
2. If chosen before training, call it a hyperparameter.
3. If learned from the data, call it a parameter.

**Likely exam questions**:

* "In K means, which is the hyperparameter and which are the learned parameters?"
* "Why is batch size a hyperparameter?"

## 4. Design Questions

### 66. You are designing a disease detection system where positive cases are rare. Which evaluation measures should you emphasize, and why?

**Answer**: Emphasize recall, precision, and F score rather than accuracy. Accuracy can be misleading in an imbalanced dataset. Recall is especially important if missing a true disease case is costly. Precision is also useful because too many false alarms can create unnecessary follow up work.

### 67. You are designing a fraud detector and the dataset is heavily imbalanced. What is the simplest data level fix you would try first?

**Answer**: Start with resampling. I would consider random undersampling of the majority class or oversampling of the minority class. If the features are numeric, SMOTE is also a reasonable next option.

### 68. You have a multiclass dataset with one very large class and two small classes. How would you handle the imbalance?

**Answer**: Treat the class sizes as a hyperparameter design choice. Pick a reference class size, then oversample or undersample the others relative to it. After that, compare model performance to see whether the new balance helps.

### 69. You need to generate more minority samples, but the data are raw face images. Would you apply SMOTE directly to the pixels?

**Answer**: No, not directly. SMOTE works best in numeric feature spaces where interpolation is meaningful. Direct pixel interpolation usually creates unrealistic images. A better design is to first embed the images into a more meaningful representation, then apply interpolation there.

### 70. You are given only normal machine-operation data and almost no labeled anomalies. Which anomaly detection method from this lecture is a strong fit, and why?

**Answer**: One class SVM is a strong fit because it learns a boundary around normal data without needing labeled anomaly classes. An autoencoder trained on normal data is also a reasonable choice if reconstruction error can separate normal from abnormal patterns.

### 71. You are building an anomaly detector for noisy sensor data, and you also want to remove measurement noise. Which method from the lecture is especially suitable?

**Answer**: A denoising autoencoder is especially suitable. It can learn to reconstruct the clean version of noisy input data, and reconstruction error can also help identify anomalous samples.

### 72. You are choosing between one decision tree and an ensemble of many classifiers. Accuracy is important, but so is explanation. What tradeoff should you mention?

**Answer**: An ensemble may improve predictive performance, but explanation becomes weaker. A single decision tree or stump is easier to explain because its reasoning is more transparent. An ensemble is often more accurate, but the final decision is harder to justify clearly to an end user.

### 73. You need an ensemble method, but the system must train quickly in parallel on multiple machines. Would you prefer bagging or boosting?

**Answer**: Prefer bagging. Its bootstrap samples and base classifiers can be trained independently, which makes it easier to parallelize. Boosting is sequential because each round depends on the updated weights from the previous round.

### 74. You need an ensemble that focuses more on difficult training cases than easy ones. Which method is the better choice?

**Answer**: Boosting is the better choice. It increases the weights of misclassified samples so later classifiers focus more on difficult cases.

### 75. You want a tree based ensemble that reduces correlation between trees. Which method fits best, and what design feature gives it that benefit?

**Answer**: Random forest fits best. It reduces correlation by not only using bootstrap samples, but also by choosing only a random subset of features at each split.

### 76. You have a very large pool of classifiers and want to combine only a promising subset. What design stages should your answer mention?

**Answer**: Mention classifier generation, classifier selection, and classifier combination. For selection, you can discuss forward selection, backward elimination, heuristics, or metaheuristic search.

### 77. You are asked to design an ensemble for a pool where some classifiers are clearly stronger than others. How should you combine them?

**Answer**: Use weighted majority voting rather than simple majority voting. Stronger classifiers should have more influence on the final decision.

### 78. You are designing a system for market basket data and want to discover which products tend to appear together. Is this a classification task?

**Answer**: No. This is an association rule mining task. The goal is to discover item relationships, such as X implies Y, rather than predict a class label.

### 79. You need to design an efficient frequent-itemset mining method. What key principle should you rely on?

**Answer**: Use the Apriori principle. If an itemset is infrequent, all supersets containing it can be pruned. This sharply reduces the search space.

### 80. You find two products that always appear together, but they are sold only a few times overall. What limitation of the framework should you mention?

**Answer**: A minimum support based framework may miss that rule. Even if confidence is high, the pattern can disappear if total support is below the minimum support threshold.

### 81. You are asked to justify why you chose a particular batch size or another training setting in a project report. What kind of answer did the lecturer suggest is acceptable?

**Answer**: It is acceptable to justify the choice by saying it follows common practice in the literature or widely used best practices, as long as the setting is reasonable for the problem.

### 82. A model performs extremely well on training data but poorly on test data. What redesign steps would be reasonable based on this lecture?

**Answer**: Reduce effective model complexity, add more data if possible, or use regularization methods such as weight penalties or dropout. These are reasonable ways to address overfitting.

### 83. A model performs poorly on both training and test data. What redesign explanation fits this pattern?

**Answer**: This suggests underfitting. The model may be too simple, insufficiently trained, or using features that do not carry enough useful signal.

## 5. Best Topics To Prioritize

If the exam avoids heavy calculation, these are the highest yield topics to know well:

* Why accuracy fails in class imbalance
* Precision, recall, specificity, false positive rate, and false negative rate
* ROC and AUC meaning
* Resampling and SMOTE
* Outlier versus noise
* Autoencoder uses
* One class SVM interpretation
* Ensemble diversity and voting
* Bagging versus boosting
* Random forest idea
* Support, confidence, minsup, and minconf
* Apriori principle and pruning
* Hyperparameter versus parameter
