# Titanic Survival Prediction Analysis

This repository contains a **data analysis project** focused on predicting passenger survival aboard the **RMS Titanic**, based on the [Kaggle Titanic competition dataset](https://www.kaggle.com/competitions/titanic/data).  
The analysis demonstrates **data preprocessing, model building (Logistic Regression)**, and **prediction generation** using both **Python** and **R**, each containerized with **Docker** for full reproducibility.

---

## Project Overview

The goal of this project is to analyze Titanic passenger data (`train.csv`) and build a predictive model to estimate survival probabilities.  
The workflow includes:

1. **Loading the datasets** (`train.csv` and `test.csv`).
2. **Preprocessing** — handling missing values and encoding categorical variables.
3. **Training** a Logistic Regression model using selected features.
4. **Evaluating** the model’s accuracy on the training data.
5. **Generating predictions** for the test dataset.

The complete process is implemented **twice**:
- once in **Python**, and  
- once in **R**,  

each inside a **Docker container** to ensure consistent environments.

---

## 🗂️ Project Structure

```
.
├── .gitignore               # Specifies intentionally untracked files
├── README.md                # This file
└── src/
    ├── data/                # Data files (must be downloaded manually)
    │   ├── train.csv        # ---> PLACE DOWNLOADED FILE HERE <---
    │   └── test.csv         # ---> PLACE DOWNLOADED FILE HERE <---
    ├── python_analysis/     # Python implementation
    │   ├── Dockerfile
    │   ├── main.py
    │   └── requirements.txt
    └── r_analysis/          # R implementation
        ├── Dockerfile
        ├── install_packages.R
        └── main.R
```

---

## Setup Instructions

### 1. Prerequisites

Before running this project, ensure you have the following installed:

- **[Git](https://git-scm.com/downloads)** — to clone the repository  
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** — to build and run the analysis containers  

---

### 2. Clone the Repository

```bash
git clone https://github.com/QuentinKunYu/titanic-disaster.git
cd titanic-disaster
```

---

### 3. Download the Data

Download the dataset files from Kaggle:

🔗 [Kaggle Titanic Competition Data](https://www.kaggle.com/competitions/titanic/data)

Place the following files inside the `src/data/` directory:

```
src/data/train.csv
src/data/test.csv
```

> **Note:** The `src/data` directory is listed in `.gitignore`, so these data files will not be tracked by Git.

---

## Running the Analysis

Make sure **Docker Desktop** is running before executing the following commands.  
All commands should be run from the **root directory** (`titanic-disaster`).

---

###  1: Python Analysis

#### Build the Python Docker Image
```bash
docker build -t titanic-python -f src/python_analysis/Dockerfile .
```

#### Run the Python Container
```bash
docker run -v ./src/data:/app/src/data titanic-python
``` 

---

###  2: R Analysis

#### Build the R Docker Image
```bash
docker build -t titanic-r -f src/r_analysis/Dockerfile .
```

#### Run the R Container
```bash
docker run -v ./src/data:/app/src/data titanic-r
```

---


