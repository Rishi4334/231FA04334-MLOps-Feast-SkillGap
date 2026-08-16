# 231FA04334-MLOps-Feast-SkillGap

## Student Details

**Name:** P RISHI SARAN
**Register Number:** 231FA04334
**Section:** 15

---

# Problem Statement

## Curriculum-Industry Skill Alignment Decision Framework for Identifying Employability Skill Gaps in CSE Graduates

The objective of this project is to develop a machine learning-based framework that identifies employability skill gaps among Computer Science and Engineering (CSE) graduates.

The dataset contains academic performance, programming skills, technical skills, soft skills, cloud and DevOps skills, internships, projects, certifications, interview performance, and other employability-related attributes.

The system uses these features to predict the student's **Skill Gap Level**. The project also demonstrates the use of **Feast Feature Store** to consistently manage and retrieve machine learning features during model training and prediction.

---

# Dataset

## Dataset Description

The dataset represents CSE students and their academic, technical, professional, and employability-related characteristics.

The dataset contains **100+ columns** covering multiple categories of student information.

### Main Dataset Columns

The dataset includes the following columns:

```text
Student_ID
Age
Gender
Degree
Specialization
Current_Year
Semester
CGPA
Attendance_Percentage
Internal_Marks
External_Marks
Mini_Project_Score
Major_Project_Score
Lab_Performance
Assignment_Average
Backlogs
Academic_Consistency
Class_Rank

C_Programming
C++_Programming
Java
Python
JavaScript
SQL
HTML_CSS
ReactJS
NodeJS
Programming_Logic
Arrays
Linked_Lists
Stacks_Queues
Trees
Graphs
Dynamic_Programming
Searching_Sorting
Competitive_Coding

Statistics
Linear_Algebra
Machine_Learning
Deep_Learning
NLP
Computer_Vision
Data_Preprocessing
Feature_Engineering
Model_Evaluation
Data_Visualization

AWS
Azure
Google_Cloud
Docker
Kubernetes
MLflow
DVC
FastAPI
CI_CD
Model_Monitoring

MySQL
PostgreSQL
MongoDB
Firebase
REST_API
Spring_Boot
ExpressJS
API_Development

Git
GitHub
Agile
Jira
Linux
VSCode_Productivity

Computer_Networks
Operating_Systems
Cybersecurity
Network_Security
Ethical_Hacking
Linux_Administration

Communication
Teamwork
Leadership
Problem_Solving
Critical_Thinking
Time_Management
Adaptability
Creativity
Presentation_Skills
English_Proficiency

Internship_Count
Internship_Duration_Months
Projects_Completed
Open_Source_Contributions
GitHub_Repositories
LeetCode_Problems_Solved
HackerRank_Badge_Level
Certifications
Hackathons_Attended

Resume_Score
Aptitude_Score
Technical_Interview_Score
HR_Interview_Score
Mock_Interview_Score

Skill_Gap_Level
```

## Target

The target variable is:

```text
Skill_Gap_Level
```

It represents the predicted employability skill-gap category of a student.

## How the Dataset Entries Were Created

The dataset contains student-level records representing academic performance, technical skills, professional experience, soft skills, and employability indicators.

The entries were created as a structured dataset for demonstrating curriculum-industry skill-gap analysis and machine learning. The values represent different levels of student performance and skill development across the listed attributes.

---

# Feature Engineering

The original dataset contains many raw attributes. A smaller set of meaningful features was prepared for the Feast FeatureView.

## Feast Features

| Feature                    | Meaning                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| `gender_encoded`           | Encoded student gender                                                                      |
| `age`                      | Student age                                                                                 |
| `cgpa`                     | Cumulative academic performance                                                             |
| `attendance`               | Student attendance percentage                                                               |
| `academic_score`           | Combined academic performance indicator                                                     |
| `project_score`            | Combined mini-project, major-project and lab performance                                    |
| `technical_skill_score`    | Average performance across selected technical skills                                        |
| `soft_skill_score`         | Average performance across selected soft skills                                             |
| `industry_readiness_score` | Combined indicator based on internships, projects, open-source contributions and hackathons |
| `internships`              | Number of internships completed                                                             |
| `projects`                 | Number of projects completed                                                                |

## Example Feature Calculation

One engineered feature is:

```text
academic_score =
(
    CGPA
    + Attendance_Percentage / 10
    + Internal_Marks / 10
    + External_Marks / 10
) / 4
```

This combines several academic indicators into a single feature representing overall academic performance.

Another example is:

```text
project_score =
(
    Mini_Project_Score
    + Major_Project_Score
    + Lab_Performance
) / 3
```

This provides a combined measure of practical/project performance.

---

# Feast Architecture

```text
Original Dataset
      ↓
Feature Engineering
      ↓
Parquet Offline Data
      ↓
Feast FeatureView
      ↓
 ┌─────────────────────┐
 ↓                     ↓
Historical Features   Materialization
 ↓                     ↓
Model Training       Online Store
                       ↓
                  Online Retrieval
                       ↓
                    Prediction
```

---

# Feast Implementation

## 1. Entity

The entity in the Feast implementation is:

```text
student
```

The join key is:

```text
student_id
```

The entity uniquely identifies an individual student whose features are stored and retrieved from Feast.

---

## 2. Data Source

The processed feature data is stored as a Parquet file:

```text
data/cse_employability_features.parquet
```

The Parquet file contains the engineered features along with:

```text
student_id
event_timestamp
created_timestamp
```

The `event_timestamp` allows Feast to associate feature values with a point in time.

---

# 3. FeatureView

The FeatureView is:

```text
cse_employability_features
```

It contains the following features:

```text
gender_encoded
age
cgpa
attendance
academic_score
project_score
technical_skill_score
soft_skill_score
industry_readiness_score
internships
projects
```

The FeatureView connects the student entity, feature definitions, and Parquet data source.

---

# 4. Historical Feature Retrieval

Historical features are retrieved from the offline store using the event timestamps.

Historical feature retrieval is useful for machine learning training because the model should receive the feature values that would have been available at the relevant point in time.

This helps avoid inconsistent feature calculations between training and prediction.

---

# 5. Model

A **Decision Tree Classifier** was used as the initial machine learning model.

The model was configured as:

```python
DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
```

The dataset was divided into:

```text
80% Training Data
20% Testing Data
```

using:

```python
train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The model predicts:

```text
Skill_Gap_Level
```

---

# 6. Online Retrieval

After materialization, features can be retrieved from the Feast online store using the student's `student_id`.

Example:

```python
online_features = store.get_online_features(
    features=feature_service,
    entity_rows=[
        {"student_id": "STU000001"}
    ]
).to_dict()
```

The retrieved features are converted into a DataFrame and supplied to the trained machine learning model.

---

# Answers to the Required Questions

## 1. What is the entity in your Feast implementation?

The entity is **student**.

The entity uses:

```text
student_id
```

as its join key.

It uniquely identifies each student whose features are managed by Feast.

---

## 2. List the features stored in your FeatureView.

The FeatureView stores:

```text
gender_encoded
age
cgpa
attendance
academic_score
project_score
technical_skill_score
soft_skill_score
industry_readiness_score
internships
projects
```

---

## 3. Explain how one feature was calculated.

The `technical_skill_score` was calculated as the average of selected technical skill features.

For example:

```text
technical_skill_score =
mean(
    C_Programming,
    C++_Programming,
    Java,
    Python,
    JavaScript,
    SQL,
    HTML_CSS,
    ReactJS,
    NodeJS,
    Programming_Logic,
    Arrays,
    Linked_Lists,
    Stacks_Queues,
    Trees,
    Graphs,
    Dynamic_Programming,
    Searching_Sorting,
    Competitive_Coding,
    Machine_Learning,
    Data_Preprocessing,
    Feature_Engineering,
    Model_Evaluation
)
```

This converts multiple technical skill measurements into a single overall technical skill indicator.

---

## 4. What is the difference between your original dataset and the feature dataset?

The **original dataset** contains the complete raw student information, including academic, programming, technical, cloud, soft-skill, internship, project, interview, and other attributes.

The **feature dataset** is a smaller processed dataset containing only the features required by the machine learning model and Feast.

The feature dataset also contains:

```text
student_id
event_timestamp
created_timestamp
```

These are required for Feast entity identification and time-aware feature management.

Therefore:

```text
Original Dataset
= Raw + many student attributes

Feature Dataset
= Selected + engineered ML features + Feast timestamps
```

---

## 5. What is the purpose of the offline store?

The offline store contains historical feature data.

It is mainly used for:

* Training machine learning models
* Generating historical training datasets
* Performing point-in-time feature retrieval
* Maintaining historical feature values

In this implementation, the offline data is stored using Parquet files.

---

## 6. What is the purpose of the online store?

The online store provides fast access to the latest materialized feature values.

It is mainly used during model inference or prediction.

For example, when a student's `student_id` is provided, Feast can retrieve that student's stored features from the online store and provide them to the machine learning model.

The current implementation uses:

```text
SQLite
```

as the local online store.

---

## 7. What is the purpose of `feast apply`?

`feast apply` registers the Feast definitions with the feature store.

It applies the definitions contained in `features.py`, including:

* Entity
* Data Source
* FeatureView
* FeatureService

It essentially tells Feast about the structure and metadata of the features that should be managed.

---

## 8. What does materialization do?

Materialization copies feature values from the historical/offline data source into the online store for a specified time range.

For example:

```text
Offline Parquet Data
        ↓
   Materialization
        ↓
    Online Store
```

After materialization, applications can retrieve the features quickly using the student's entity key.

---

## 9. What is the advantage of retrieving features through Feast instead of manually calculating them separately during training and prediction?

The main advantage is **feature consistency**.

Without a feature store, the same feature calculations may be implemented separately during training and prediction, which can cause inconsistencies.

Feast provides a centralized definition of features that can be reused for:

```text
Training
   +
Prediction
```

This reduces the possibility of training-serving skew and makes feature management easier.

It also provides:

* Reusable features
* Historical feature retrieval
* Online feature retrieval
* Centralized feature definitions
* Consistent feature computation

---

## 10. State two limitations of your current dataset.

### Limitation 1 — Limited real-world industry evidence

The current dataset is primarily a structured student-level dataset and does not contain enough direct evidence from actual companies, job descriptions, hiring requirements, or industry skill-demand data.

### Limitation 2 — Limited temporal information

The dataset does not contain comprehensive real-world time-series information showing how student skills and industry requirements change over different academic years or hiring cycles.

The current Feast timestamps are used for feature-store demonstration and are not actual historical industry-event timestamps.

---

## 11. State two ways your feature store could be improved when more curriculum and industry evidence becomes available.

### Improvement 1 — Add real industry skill-demand features

The feature store could include features extracted from:

* Job descriptions
* Industry skill requirements
* Internship requirements
* Employer feedback
* Recruitment data
* Current technology trends

This would make the curriculum-industry comparison more realistic.

### Improvement 2 — Add time-aware curriculum and industry features

The feature store could maintain historical versions of skill requirements.

For example:

```text
2024 Industry Skills
        ↓
2025 Industry Skills
        ↓
2026 Industry Skills
```

This would allow the system to identify changing skill demands and provide more accurate and up-to-date skill-gap recommendations.

---

# Results

## Historical Feature Output

The historical feature dataset generated for Feast contains the engineered student features.

Example structure:

```text
student_id
event_timestamp
created_timestamp
gender_encoded
age
cgpa
attendance
academic_score
project_score
technical_skill_score
soft_skill_score
industry_readiness_score
internships
projects
```

**Insert the actual `training_data.head()` output from Colab here.**

---

## Model Accuracy

The initial model used was:

```text
Decision Tree Classifier
max_depth = 4
random_state = 42
```

The model was trained using an 80/20 train-test split.

**Model Accuracy: `[INSERT YOUR ACTUAL ACCURACY HERE]`**

The accuracy should be obtained using:

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Model Accuracy:", accuracy)
```

---

## Online Feature Output

The Feast online store was used to retrieve features using the student's entity key:

```text
student_id
```

Example:

```text
student_id    cgpa    attendance    technical_skill_score    ...
STU000001     ...        ...                  ...
STU000002     ...        ...                  ...
```

**Insert the actual `online_df` output from Colab here.**

---

## Final Prediction

The retrieved online features were passed to the trained Decision Tree model.

Example:

```python
online_df["predicted_skill_gap"] = (
    model.predict(
        online_df[feature_columns]
    )
)
```

The final prediction represents the student's predicted:

```text
Skill_Gap_Level
```

**Insert one actual prediction from your Colab output here.**

Example:

```text
Student ID: STU000001
Predicted Skill Gap Level: Medium
```

---

# Conclusion

This project demonstrates an MLOps-oriented approach for identifying employability skill gaps among CSE graduates.

The raw student dataset was transformed into meaningful machine learning features and stored as Parquet data. Feast was then used to define, manage, retrieve, and materialize these features.

A Decision Tree Classifier was trained using the historical features, while the Feast online store was used to retrieve features for prediction.

The architecture provides a foundation for extending the system with real curriculum data, industry job requirements, employer feedback, and continuously changing technology skill demands.

Future versions can integrate real industry evidence and time-aware features to improve the accuracy and practical usefulness of the curriculum-industry skill alignment framework.
