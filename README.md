# 🧑‍⚕️ Diabetes Prediction Challenge

<p align="center">
  <a href="https://www.kaggle.com/competitions/playground-series-s5e12">
    <img src="https://img.shields.io/badge/Kaggle-Playground%20Series%20S5E12-blue?logo=kaggle" alt="Kaggle" />
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.11-yellow?logo=python&logoColor=white" alt="Python" />
  </a>
  <a href="https://jupyter.org/">
    <img src="https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter" alt="Notebook" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
  </a>
  <a href="https://www.kaggle.com/riddymazumder/competitions">
    <img src="https://img.shields.io/badge/Public%20Score-0.69991-lightgrey" alt="Public Score" />
  </a>
  <a href="https://www.kaggle.com/riddymazumder/competitions">
    <img src="https://img.shields.io/badge/Private%20Score-0.69777-lightgrey" alt="Private Score" />
  </a>
  <a href="https://www.kaggle.com/competitions/playground-series-s5e12/overview">
    <img src="https://github.com/RiddyMazumder/Diabetes-Prediction-Challenge-Playground-Series---Season-5-Episode-12/blob/main/header_4.jpg" />
  </a>
</p>
Predicting the probability that a patient will be diagnosed with **diabetes** using **machine learning ensemble models**, including **CatBoost** and **LightGBM** with **optimized blending (0.60 CatBoost / 0.40 LightGBM)**.

---

## 📌 Project Overview & 🏆 Kaggle Competition

| 📌 **Project Overview** | 🏆 **Kaggle Competition & Score** |
|------------------------|----------------------------------|
| 📝 End-to-end ML project covering **EDA**, **data preprocessing**, **feature engineering**, **model training**, and **evaluation** to predict diabetes diagnosis probability. Models include **CatBoost**, **LightGBM**, and **optimized blending with threshold tuning**. | 🚀 **Diabetes Prediction Challenge – Playground Series: Season 5, Episode 12** <br> 🔗 https://www.kaggle.com/competitions/playground-series-s5e12 <br> 📊 **Public Score:** 0.69991 <br> 📊 **Private Score:** 0.69777 (**Top 21%**, 834 out of 4,206 teams) |

---

## 👤 Author

| 👤 **Name** | 🔗 **Github-Profile** |🔗 **Kaggle-Profile** |
|------------|----------------|----------------|
| Riddy Mazumder | [![GitHub](https://img.shields.io/badge/GitHub-RiddyMazumder-black?logo=github)](https://github.com/RiddyMazumder)|[![Kaggle Profile](https://img.shields.io/badge/Kaggle-RiddyMazumder-blue?logo=kaggle)](https://www.kaggle.com/riddymazumder)|

---

## 🛠️ Tools & Libraries

| 🔧 **Category** | 🛠️ **Libraries / Tools** |
|---------------|------------------------|
| Data Manipulation & Analysis | `pandas`, `numpy` |
| Data Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `catboost`, `lightgbm`, `sklearn` (`StratifiedKFold`, `accuracy_score`, `roc_auc_score`) |
| Development | `Jupyter Notebook`, `.py` scripts, `.html` exports |

---

## 🔍 Workflow & Methodology

| Step | Description |
|------|-------------|
| **1. Load Dataset** | Load patient data and inspect basic statistics using `pandas`. |
| **2. Data Exploration & Cleaning** | Visualize distributions, correlations, missing values, and outliers using `seaborn` and `matplotlib`. |
| **3. Feature Engineering** | Encode categorical variables, normalize numerical features, and create interaction features. |
| **4. Model Building** | Train base models (**CatBoost** and **LightGBM**) using **Stratified K-Fold cross-validation**. |
| **5. 🔥 Optimized Blended Predictions** | Combine CatBoost and LightGBM predictions with **blend weight 0.60 / 0.40** and **optimized threshold**, improving accuracy and AUC. |
| **6. Model Evaluation** | Compute **accuracy** and **ROC-AUC** on out-of-fold predictions. |
| **7. Conclusion & Insights** | Identify key features contributing to diabetes diagnosis probability and highlight model performance. |

---

## 📈 Model Performance

> ⭐ **Optimized Blend Performance Highlight**  
> Blending CatBoost and LightGBM predictions with tuned weight (`0.60` CatBoost / `0.40` LightGBM) and threshold (`0.500`) maximizes accuracy and overall AUC.

| Metric | Value |
|--------|-------|
| Algorithms | CatBoost, LightGBM, Optimized Blend |
| Optimized Blend Weight | 0.60 (CatBoost) / 0.40 (LightGBM) |
| Optimized Threshold | 0.500 |
| Accuracy | 0.843 (**Top 21%**, 834 out of 4,206 teams) |
| ROC-AUC | 0.91200 |
| Public Score | 0.69991 |
| Private Score | 0.69777 |
| Visualizations | Feature importance plots, prediction probability distributions |

---

## 🖥️ Mini Preview: Sample Code & Plots

```python
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

# Optimized blend of CatBoost and LightGBM predictions
best_acc = 0
best_w = 0.60  # CatBoost
best_thresh = 0.5

for w in [0.60]:  # CatBoost weight fixed
    blended = w * oof_pred_cat + (1-w) * oof_pred_lgb  # LightGBM weight = 0.40
    for thresh in np.arange(0.3, 0.7, 0.001):
        acc = accuracy_score(oof_target, (blended > thresh).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_thresh = thresh

oof_final = (best_w * oof_pred_cat + (1-best_w) * oof_pred_lgb > best_thresh).astype(int)

print(f"Optimized Blend Weight: {best_w:.2f} CatBoost / 0.40 LightGBM")
print(f"Optimized Threshold: {best_thresh:.3f}")
print(f"Optimized Accuracy: {best_acc:.5f}")
print(f"Overall AUC: {roc_auc_score(oof_target, best_w*oof_pred_cat + (1-best_w)*oof_pred_lgb):.5f}")

```
# 🔮 Future Improvements
| Improvement           | Description                                                                        |
| --------------------- | ---------------------------------------------------------------------------------- |
| Hyperparameter Tuning | Fine-tune CatBoost and LightGBM for better generalization.                         |
| Feature Engineering   | Add medical history, lab results, and derived features to improve predictions.     |
| Ensemble Methods      | Test stacking/blending with additional classifiers.                                |
| Explainability        | Use **SHAP** or **LIME** to interpret feature contributions for clinical insights. |

# 📖 References
| Resource               | Link                                                                                                                       |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Kaggle Competition     | [https://www.kaggle.com/competitions/playground-series-s5e12](https://www.kaggle.com/competitions/playground-series-s5e12) |
| CatBoost Documentation | [https://catboost.ai/docs/](https://catboost.ai/docs/)                                                                     |
| LightGBM Documentation | [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)                                                       |
| Seaborn Library        | [https://seaborn.pydata.org/](https://seaborn.pydata.org/)                                                                 |
## 📄 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project for educational and personal purposes.

---

## 💬 Feedback & Contributions

If you have any questions, suggestions, or improvements:

* 🐞 Open an issue
* 📤 Submit a pull request

Contributions are always welcome! 😊

---

Happy analyzing and learning! 🚀
