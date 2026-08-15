# data-beats-algorithms
# 📈 Data Beats Algorithms: Financial Crisis Predictor

**Live Dashboard:** [Link to your Streamlit App here once deployed]

## 🚀 Overview
An empirical machine learning proof using 150 years of macroeconomic data to demonstrate the Golden Rule of ML: **Data Preprocessing > Algorithm Choice.**

This project is an advanced machine learning pipeline designed to predict systemic financial crises using historical macroeconomic data across multiple global economies. 

## 📜 Certification & Validation
This advanced predictive model was engineered to practically apply and expand upon the core machine learning and data science methodologies I have been certified in. 

<div align="center">
  <img src="certificate.png" alt="Certificate of Achievement" width="80%">
</div>

---

## 🏆 The Empirical Proof: Algorithm vs. Preprocessing

During the development of this predictor, 7 different machine learning algorithms were raced head-to-head on the same 150-year dataset. The experiment yielded a profound discovery regarding signal processing.

<table>
  <tr>
    <th align="center">❌ BEFORE: Unsorted Data Pipeline</th>
    <th align="center">✅ AFTER: Strictly Sorted & Grouped Pipeline</th>
  </tr>
  <tr>
    <td align="center"><img src="LSTM%20winner.png" alt="LSTM Winning on Noisy Data" width="100%"></td>
    <td align="center"><img src="XGBOOST%20winner.png" alt="XGBoost Winning on Clean Data" width="100%"></td>
  </tr>
  <tr>
    <td><b>Winner: LSTM Network</b><br><i>Hint: The HP Filter was used here too!</i> However, because the data lacked strict chronological sorting by country and relied on basic differencing, the HP Filter processed noisy, blended timelines. Deep neural networks won by attempting to "guess" through this structural noise.</td>
    <td><b>Winner: XGBoost</b><br>When the HP Filter was applied <i>correctly</i> (strictly grouped by country and sorted chronologically), combined with true second-derivative acceleration (`.diff().diff()`), the fog cleared. With pristine signals, XGBoost completely dominated the leaderboard.</td>
  </tr>
</table>

### The Takeaway
Both tests utilized the Hodrick-Prescott (HP) filter, but the application made all the difference. By mathematically isolating the true economic cycles strictly within individual country timelines and measuring the true *acceleration* of debt, the data became pristine. Once the data was correctly structured, **XGBoost** proved to be the most accurate and precise engine for predicting rare financial crises. 

*(Note: The original noisy pipeline testing can be found preserved in the `experiments/` folder for reproducibility).*

## ⚙️ The Preprocessing & Engineering Pipeline

To handle centuries of extreme economic shifts, the winning pipeline utilizes:
1. **Chronological State Isolation:** Data is strictly grouped by country and year to prevent timeline bleeding.
2. **Hodrick-Prescott (HP) Filtering:** Applied natively via `statsmodels` ($\lambda = 100$) to separate standard economic growth from dangerous, volatile credit bubbles.
3. **Credit Acceleration:** Using second derivatives (`.diff().diff()`) to measure the exact moment debt begins compounding out of control.
4. **Robust Scaling:** `RobustScaler` is deployed to prevent massive historical outliers (e.g., hyperinflation, world wars) from breaking the decision boundaries.

## 🛠️ Tech Stack
* **Core Machine Learning:** XGBoost, Scikit-Learn
* **Econometrics & Signal Processing:** Statsmodels
* **Data Manipulation:** Pandas, NumPy
* **Frontend:** Streamlit

## 💻 How to Run Locally

**1. Clone the repository and navigate to the project folder:**
```bash
git clone [https://github.com/Sanju-1001011/data-beats-algorithms.git](https://github.com/Sanju-1001011/data-beats-algorithms.git)
cd data-beats-algorithms/"file 2"

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the interactive dashboard:**
   ```bash
   python -m streamlit run app.py
   ```
