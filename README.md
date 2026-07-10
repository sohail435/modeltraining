---
title: SaaS Churn and Retention Engine
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
---

## SaaS Churn & Retention Engine Portfolio Piece
Automated MLOps evaluation and deployment pipeline.# 📊 SaaS Customer Churn & Retention Intelligence Engine (v1.3.0)

A production-grade machine learning micro-application designed to evaluate customer telemetry, predict account churn risk probabilities, and isolate behavioral risk factors before cancellations occur.

## 🚀 Key Architectural Engineering Highlights

*   **Bayesian Hyperparameter Optimization:** Implements `Optuna` to intelligently navigate regularized search fields and isolate optimal weights, boosting minority class identification efficiency.
*   **Pipeline Data Isolation:** Utilizes a unified Scikit-Learn `Pipeline` and `ColumnTransformer` architecture to handle scaling and encoding concurrently, completely eliminating data leakage risks.
*   **Cost-Sensitive Learning (Class Balancing):** Addresses extreme class imbalance using automated sample weighting penalties, increasing **Class 1 (Churn) Recall from 20% to 86%** to capture the vast majority of at-risk users.
*   **Responsive Enterprise Dashboard:** Built with a custom dark-slate layout that adapts dynamically to both mobile and desktop screens, featuring automated batch CSV fleet processing and real-time interactive evaluation cards.

---

## 🛠️ Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/sohail435/modeltraining.git](https://github.com/sohail435/modeltraining.git)
   cd modeltraining
