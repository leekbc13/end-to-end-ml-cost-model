
---
# End-to-End Machine Learning Pipeline – Cost Prediction Project

This project demonstrates a complete machine learning pipeline built to predict individual costs using demographic and lifestyle variables.
It was completed as part of a graduate-level **Big Data** course at Boston College.

<!-- Optional banner image -->
<!-- ![Project Overview](plots/pipeline-preview.png) -->

## 🔍 Project Overview

We developed an end-to-end solution that:
- Cleans and preprocesses raw input data
- Performs exploratory data analysis (EDA)
- Trains and saves a predictive model using Scikit-learn
- Evaluates model performance and outputs cost predictions on new data

The workflow simulates a real-world deployment pipeline, showing how raw inputs can be transformed into actionable insights.

## 🧠 Tools & Technologies

- **Python 3.9**
- **Pandas**, **NumPy**, **Scikit-learn**
- **Joblib** (model serialization)
- **CSV** for test predictions
- Jupyter-compatible `.py` scripts

## 🧑‍🤝‍🧑 Collaboration Note

This project was developed collaboratively with peers **Gina Occhipinti**, **Reuben Vincent**, and **Kache Lee**.  
My contributions included:
- Leading the model development
- Structuring and cleaning the dataset
- Building the final pipeline scripts
- Organizing the project repository

## 📁 Project Structure

```

end-to-end-ml-cost-model/
├── code/
│   ├── 01\_data\_analysis.py
│   ├── 02\_data\_cleaning.py
│   └── 03\_train\_and\_save\_model.py
├── model/
│   └── final\_model.pkl
├── data/
│   └── excel\_test\_row\_with\_prediction.csv
├── .gitignore
└── README.md

````

## 📈 Key Takeaways

- Demonstrated the ability to build a fully functioning ML workflow from scratch
- Emphasized reproducibility, modular code, and team collaboration
- Gained hands-on experience with model persistence and deployment simulation

---

## 🚀 How to Run This Project

1. Clone the repo:
   ```bash
   git clone https://github.com/leekbc13/end-to-end-ml-cost-model.git
   cd end-to-end-ml-cost-model
````

2. Run the cleaning and model scripts:
   ```bash
   python code/02_data_cleaning.py
   python code/03_train_and_save_model.py
````

---

## 📬 Contact

For questions about this project or collaboration opportunities, feel free to connect:
**Kache Lee**
[GitHub](https://github.com/leekbc13) | [LinkedIn](https://www.linkedin.com/in/kachelee)



