# House Price Prediction with Genetic Algorithm Optimization

Comparing 3 regression models (KNN, Decision Tree, Linear Regression) for house price prediction, with Genetic Algorithm performing feature selection and hyperparameter tuning jointly.

**Course Project** — Dasar Kecerdasan Artifisial, Telkom University 2025

**Team:** Dimas Muhammad Akbar, Muhamad Iqbal Tsany Putra, Arya Danuharja

---

## Problem

Standard hyperparameter tuning (GridSearch, RandomSearch) treats feature selection and hyperparameter optimization as separate problems. This project uses a Genetic Algorithm to optimize both **simultaneously**, encoding feature mask and model hyperparameters in a single chromosome.

## Approach

- **Dataset:** Housing dataset with 1,000 rows × 10+ features
- **Models compared:** K-Nearest Neighbors, Decision Tree, Linear Regression
- **GA implementation:** Built from scratch with NumPy
  - Population: 30, Generations: 15
  - Tournament selection (k=3), Uniform crossover (p=0.8)
  - Bit-flip mutation (0.03) for feature mask, Integer mutation (0.25) for hyperparameters
- **Evaluation:** Train/Validation/Test split, RMSE and R² as primary metrics

## Results

| Model | Test RMSE | Test R² | Train-Val RMSE Gap |
|-------|-----------|---------|---------------------|
| **Linear Regression + GA** | **$9,960** | **0.9986** | -632 (no overfitting) |
| Decision Tree + GA | $27,181 | 0.9897 | 8,182 |
| KNN + GA | $29,995 | 0.9874 | 29,717 (overfit) |

## Key Findings

1. **Simpler models won.** Linear Regression + GA outperformed tree-based models because the underlying data relationship is near-linear (Square_Footage correlation 0.991).
2. **GA-based feature selection helps prevent overfitting in linear models** but does not save KNN from severe overfitting on this dataset.
3. **Joint optimization** of features + hyperparameters is more efficient than sequential tuning when you have a clear single fitness signal.

## Tech Stack

- Python 3.x, NumPy, pandas
- scikit-learn (model implementations)
- matplotlib, seaborn (visualization)
- Jupyter Notebook

## Repository Structure

- `house_price_regression_ga.ipynb` — Main notebook
- `README.md`
- `.gitignore`

## How to Run

1. Open `house_price_regression_ga.ipynb` in Jupyter or Google Colab
2. Install dependencies: `pip install numpy pandas scikit-learn matplotlib seaborn`
3. Run cells sequentially

## My Contribution (Muhamad Iqbal Tsany Putra)

- Sourced and selected the dataset
- Defined the project scope and modeling approach
- Implemented the Genetic Algorithm in NumPy, chromosome encoding (feature mask + hyperparameters), tournament selection, uniform crossover, and bit-flip / integer mutation operators
- Built the full training and evaluation pipeline for all 3 models (KNN, Decision Tree, Linear Regression)
- Performed overfitting diagnosis via train/validation/test split analysis and authored the comparative results
