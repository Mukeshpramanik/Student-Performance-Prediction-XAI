# Dataset Documentation

## Source
- **Origin**: UCI Machine Learning Repository
- **Dataset**: Student Performance (Mathematics course)
- **Primary Reference**: P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*, Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), Porto, Portugal, pp. 5-12, 2008.
- **License**: CC BY 4.0

## Contents
- `raw/student-mat.csv`: Unmodified raw dataset containing 395 student observations across 33 variables, delimited with semicolons.
- `processed/student_clean.csv`: Cleaned dataset with validated types and at-risk binary indicator (`at_risk` = G3 < 10).

## Feature Categories
- **Demographic**: `sex`, `age`, `address`, `famsize`, `Pstatus`
- **Family Background**: `Medu`, `Fedu`, `Mjob`, `Fjob`, `guardian`
- **Academic Context**: `school`, `reason`, `traveltime`, `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`
- **Social & Health Factors**: `romantic`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`
- **Grades**: `G1` (first period grade: 0-20), `G2` (second period grade: 0-20), `G3` (final evaluation grade: 0-20, target)
