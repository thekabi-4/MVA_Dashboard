# 🎓 MVA Academic Performance Analyzer – 10th Grade Dashboard

> **Insightful, automated analysis of 10th-grade academic results with AI-powered clustering, outlier detection, and interactive visualizations.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?logo=scikit-learn)
![MySQL](https://img.shields.io/badge/MySQL-8.0%2B-4479A1?logo=mysql)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Overview

The **MVA Academic Performance Analyzer** is an end-to-end data science pipeline designed to transform raw 10th-grade marksheet data (from Excel) into **actionable educational insights**. Built for schools and administrators, it automates performance evaluation, identifies at-risk students, compares cohorts, and uncovers hidden patterns using machine learning — all through an intuitive **Streamlit dashboard**.

Originally developed for **MVA School**, this tool supports standardized Indian grading systems (CBSE/State Board) and handles real-world data inconsistencies with robust parsing.

---

## 🔍 Key Features

### 📊 **Interactive Dashboard**
- Subject-wise performance trends
- Grade distribution (A1, A2, B1, ..., E) with **color-coded pie charts**
- Correlation heatmap + **plain-language insights** (e.g., "Maths and Science are strongly correlated")
- Outlier detection via **IQR, Z-score, and Isolation Forest**
- Student clustering into performance groups (K-Means + Silhouette analysis)

### 🤖 **Advanced Analytics**
- **Linear regression** to predict overall percentage from subject scores
- **PCA** for dimensionality reduction and pattern visualization
- **T-test** to statistically compare "New" vs "Old" student cohorts
- Automatic detection of data duplicates using **MD5 hashing**

### 💾 **Data Management**
- Parses messy Excel sheets with flexible header detection
- Saves all results to **MySQL** (students, models, analysis runs)
- Reusable: Load past analyses without reprocessing

---

## 🛠️ Tech Stack

- **Core**: Python, Pandas, NumPy
- **ML**: scikit-learn (LinearRegression, KMeans, IsolationForest, PCA)
- **Stats**: SciPy (t-tests, z-scores)
- **DB**: SQLAlchemy + MySQL
- **Frontend**: Streamlit + Plotly (responsive, publication-ready charts)
- **Utils**: hashlib, joblib, datetime

---

## 📁 Project Structure

```
MVA_Dashboard/
├── app.py                  # Streamlit entry point
├── data_processor_10th.py  # Core logic: parsing, analysis, dashboard
├── sample_data/            # Example marksheet (Excel)
└── requirements.txt        # Dependencies
```

---

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up MySQL** (optional but recommended):
   - Update `db_config` in your `app.py`
   - The script auto-creates the database and tables

3. **Run the dashboard**:
   ```bash
   streamlit run app.py
   ```

4. **Upload** your 10th-grade Excel marksheet and explore!

> 💡 No Excel file? Use the included `sample_10th_marksheet.xlsx` to test.

---

## 📈 Sample Insights

| Insight Type | Example |
|-------------|--------|
| **Performance** | "Average Science score: 78.2 (B1 grade)" |
| **Correlation** | "Maths & Science: r = 0.82 → Strong positive" |
| **Outliers** | "3 students scored >95% — potential toppers" |
| **Clustering** | "Cluster 0: High performers in all subjects" |
| **Risk Alert** | "5 students in 'E' grade — need intervention" |

---

## 🤝 Contributing

Contributions are welcome!  
- 🐛 Found a bug? Open an issue.
- 💡 Have an idea? Submit a feature request.
- ✨ Want to improve? Fork and send a PR.

---

## 📜 License

MIT License — free to use, modify, and distribute for educational or commercial purposes.

---

## 🙏 Acknowledgements

Built with ❤️ for educators and students.  
Inspired by the need for **data-driven decision-making in Indian schools**.

---

> **Empower teachers. Support students. Understand performance.**  
> — *MVA Academic Performance Analyzer*

---

### ✅ Ready to deploy in your school?  
Just upload the marksheet — the rest is automated.

---

