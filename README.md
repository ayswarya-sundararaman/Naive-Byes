# Naive Bayes Classification - DonorsChoose Dataset

This project implements **Multinomial Naive Bayes (NB)** for predicting project approval on the DonorsChoose dataset using bag-of-words (BOW) and TF-IDF features.

## Dataset
- **Dataset**: DonorsChoose project proposals.
- **Target**: Binary classification for project approval (1: approved, 0: not approved).

## Key Features
- **Text Features**: Preprocessed project essays, titles using BOW and TF-IDF techniques.
- **Categorical Features**: Grade category, subject categories, teacher prefix.
- **Numerical Features**: Resource cost, number of previously submitted projects.

## Process
1. **Preprocessing**:
   - Cleaned and vectorized text data.
   - One-hot encoded categorical features.

2. **Modeling**:
   - Applied **Multinomial Naive Bayes** with hyperparameter tuning for `alpha` to optimize AUC performance.
   - Used cross-validation (k-fold) and RandomizedSearchCV for hyperparameter search.

3. **Evaluation**:
   - Plotted ROC curves and computed AUC for train and test data.
   - Identified top features using feature importance from NB's `feature_log_prob_`.

## Results
- **Best Alpha**: 0.0001
- **Best AUC on Cross-validation**: 0.77

## Conclusion
Multinomial Naive Bayes performs well on text-based features from the DonorsChoose dataset, with BOW and TF-IDF providing good predictive performance.
