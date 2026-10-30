# Machine Learning — Practical Session 1: Data Exploration and Preprocessing

**National Higher School of Advanced Technologies (ENSTA)**  
**Academic Year:** 2026–2027  
**Student:** Anes Benrahmoune  
**Level:** 4th Year Student (Artificial Intelligence Specialty)  
**Group:** Group 1  
**GitHub:** [BENRAHMOUNEANES](https://github.com/BENRAHMOUNEANES)  
**Course:** Machine Learning  

---

## 1. Project Overview & Repository Structure
This repository contains the complete implementation and report for Practical Session 1 covering data exploration, stratified data partitioning, text processing, missing data diagnostics and imputation, feature scaling, binarization, and categorical encoding.

The folder structure strictly matches the course standards:
```
tp1/
├── README.md                      # Comprehensive project documentation & discussion notes
├── requirements.txt               # Required Python packages
├── TP1_Solutions.ipynb            # Fully executed Jupyter Notebook with all outputs & plots
├── exercises/
│   └── Practical session 1.pdf    # Lab assignment sheet
└── data/
    ├── Iris_Dataset.csv           # Tabular dataset for Exercise 1
    ├── txt_file.txt               # Raw text file for Exercise 2
    └── employees.csv              # Missing values dataset for Exercise 3
```

---

## 2. Quick Start
```bash
# 1. Install required packages
pip install -r requirements.txt

# 2. Launch the Jupyter Notebook
jupyter notebook TP1_Solutions.ipynb
```

---

## 3. Summary of Exercises & Discussion Answers

### Exercise 1 — Tabular Exploration & Stratified Splitting
- **Dataset:** `Iris_Dataset.csv` (150 rows, 4 continuous features, 1 multiclass target `species`).
- **Inspection Methods:**
  - `info()`: Confirms 150 non-null values across 4 `float64` features and 1 string column. Total memory footprint: ~7.9 KB.
  - `describe()`: Quantifies distributions. Identifies that `petal_length` has high dispersion ($\sigma = 1.764$), while `sepal_width` has low dispersion ($\sigma = 0.434$).
  - `shape`: $(150, 5)$.
  - `columns`: `['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']`.
  - `head()`: Visualizes sample observations in centimeters.
- **Copying the DataFrame:** Creating a deep copy (`D.copy()`) avoids in-place mutation of ground truth and prevents pandas `SettingWithCopyWarning`.
- **Stratified Split:** Data is split 80/20 with `stratify=y` and `random_state=42`. Training set contains 120 samples and test set contains 30 samples, with an exact $33.3\%$ representation for each iris class.
- **Why keep the test set separate?** To prevent **data leakage (lookahead bias)**. If test samples influence scaling or imputation, the model indirectly learns statistical properties of unseen data, yielding artificially high validation scores that fail in production.

---

### Exercise 2 — Exploring Text Data
- **File Reading (`txt_file.txt`):** Read with UTF-8 encoding; trailing newlines removed via `rstrip('\r\n')`.
- **20 Newsgroups (`fetch_20newsgroups(subset='train')`):**
  - Training documents: **11,314**
  - Target labels: **11,314**
  - Unique classes: **20**
  - Numeric targets are integers ($0 \le y \le 19$) indexing into `target_names`.

---

### Exercise 3 — Detecting and Handling Missing Values
- **Initial State (`employees.csv`):** 1,000 rows. Initial missing counts: `First Name` (67), `Gender` (145), `Senior Management` (67), `Team` (43).
- **Custom Placeholders:** Pandas parses markers like `'-'` and `'na'` as standard string objects; `isna()` only identifies standard sentinels (`np.nan`, `None`, `<NA>`). They must be converted via `.replace(['-', 'na', 'N/A', '?'], np.nan)`.
- **Imputation Strategy:**
  - Categorical columns (`Gender`, `Senior Management`, `Team`): `SimpleImputer(strategy='most_frequent')`.
  - Numerical continuous columns (`Salary`, `Bonus %`): `SimpleImputer(strategy='median')` (robust to skewness and outliers).
  - Identifiers (`First Name`, `Start Date`, `Last Login Time`): **Not imputed** to prevent creating fictitious personnel records.
- **Limitations:** Imputation reduces feature variance, distorts multi-variable covariance, and masks uncertainty.

---

### Exercise 4 — Feature Scaling, Binarization, and Categorical Encoding

#### Part A: Scaling and Binarization
$$\mathbf{X} = \begin{pmatrix} 1 & -1 & 2 \\ 2 & 0 & 0 \\ 0 & 1 & -1 \end{pmatrix}$$
- `StandardScaler`: Centers mean to 0 and scales std to 1 (column-wise).
- `MinMaxScaler`: Linearly bounds values to $[0, 1]$ (column-wise).
- `MaxAbsScaler`: Divides by maximum absolute value into $[-1, 1]$ (column-wise).
- `RobustScaler`: Centers by median and scales by IQR (column-wise, outlier-resistant).
- `normalize`: Scales each row vector to unit Euclidean norm $\|x\|_2 = 1$ (row-wise / sample-wise).
- `Binarizer(threshold=0.0)`: Maps $x > 0 \to 1$ and $x \le 0 \to 0$.
- **Algorithm Sensitivity:** KNN, SVM, and Gradient Descent require scaling due to distance/margin/gradient magnitude dependency. **Decision trees are scale-invariant** because split evaluations are monotonic.

#### Part B: Categorical Encoding
- `OrdinalEncoder`: Assigns integers $0, 1, 2\dots$ to categories.
- `LabelEncoder`: Maps target categories to integer codes.
- `OneHotEncoder`: Maps categories to mutually orthogonal binary dummy vectors.
- **Why One-Hot for Nominal Features:** Nominal features have no intrinsic order. Integer encoding imposes false numerical distances ($|2 - 0| = 2$) that mislead models into learning false linear relationships.

---

## 4. Deliverable Conclusion: Preprocessing Summary

| Data Modality | Key Operations | Recommended Scikit-Learn Classes | Best Practice |
| :--- | :--- | :--- | :--- |
| **Numerical** | Imputation, Outlier Handling, Scaling | `SimpleImputer(strategy='median')`, `StandardScaler`, `RobustScaler` | Always fit transformers on training data only to avoid data leakage. |
| **Categorical** | Mode Imputation, One-Hot / Ordinal Encoding | `SimpleImputer(strategy='most_frequent')`, `OneHotEncoder`, `OrdinalEncoder` | Use One-Hot for nominal features; reserve Ordinal for ordered hierarchies. |
| **Text** | Encoding Handling, Tokenization, Cleaning, Vectorization | UTF-8 I/O, `TfidfVectorizer`, `CountVectorizer` | Standardize text, strip stopwords, and handle unseen vocabulary during test time. |
