![](images/CustomerChrun.png)

## Overview

Customer churn, also known as customer attrition, is when customer chooses to stop using the products or services of the company and stops the relationship. It is a critical metric for any business. Churn can be voluntary (customers leaving due to dissatisfaction or better alternatives etc) or involuntary (due to reasons like payment failures, fraud etc).  

The primary objective of this analysis is to identify the factors that contribute to customer churn and make a predictive model that can forecast if the customers are likely to leave 

## Dataset

The dataset used is a Telecom churn dataset (customer_churn_dataset-training-master.csv and customer_churn_dataset-testing-master.csv) from Kaggle.

Datafile Path:  

https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/data/customer_churn_dataset-training-master.csv  
https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/data/customer_churn_dataset-testing-master.csv

The datasets together contains about 500K rows and 12 feature columns including target feature:

Input Variables:

    - CustomerID: Unique identifier for a customer
    - Age: Age of the customer
    - Gender: Gender of the customer
    - Tenure: Duration in months for which a customer has been using the company's products/services
    - Usage Frequency: Number of times the customer has used the company’s services in the last month
    - Support Calls: Number of calls the customer has made to the customer support in the last month
    - Payment Delay: Number of days the customer has delayed their payment in the last month
    - Subscription Type: Type of subscription choosen by the customer
    - Contract Length: Contract duration that the customer has signed with the company
    - Total Spend: Total amount the customer has spent on the company's products or services
    - Last Interaction: Number of days since the customer had the last interaction with the company

Output Variable (Target):

    - Churn: Binary label indicating whether a customer has churned (1) or not (0) 

## Approach

1. Importing Data
2. Data cleaning
    - Clean the column names as required
    - Duplicate Check
    - Missing Value Check
    - Drop unwanted column(s)
3. Exploratory Data Analysis (EDA)
	- Outlier Treatment
	- Correlation Heatmap
	- Univariate Analysis
	- Bivariate/Multivariate Analysis
4. Data Pre-processing
	- SimpleImputer()
	- StandardScaler()
	- OneHotEncoder()
5. Modelling
	- Logistic regression (LR)
	- Decision Tree (DT)
	- Random Forest (RF)
	- K Nearest Neighbour (KNN)
	- XgBoost (XGB)
    - Support Vector Machines (SVM_SGD)
6. Model Tuning with Params
7. Comparision of Metrices
8. Evaluations
9. Recommendations and Conclusion
10. Next Steps


## Visualizations from EDA

1. [Histograms](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/Histograms.png)
2. [CorrelationHeatmap](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/CorrelationHeatmap.png)
3. [CorrelationHeatmapWRTChurn](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/CorrelationHeatmapWRTChurn.png)
4. [Distribution Of Churn](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/DistributionOfChurn.png)
5. [Distribution Of ContractLength](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/DistributionOfContractLength.png)
6. [Distribution Of Gender](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/DistributionOfGender.png)
7. [Distribution Of SubscriptionType](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/DistributionOfSubscriptionType.png)
8. [Boxplot of Features VS Churn](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/BoxplotsFeaturesVSChurn.png)
9. [PaymentDelay VS Churn](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/PaymentDelayVSChurn.png)
10. [SupportCalls VS Chrun](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/SupportCallsVSChrun.png)
11. [Subscription Type VS Churn](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/SubscriptionTypeVSChurn.png)
12. [Gender VS Churn](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/GenderVSChurn.png)
13. [KDE_X_train](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/KDE_X_train.png)
14. [KDE_X_train_scaled](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/KDE_X_train_scaled.png)
15. [ConfusionMatirx_LogisticRegression](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/ConfusionMatrix_Logistic%20Regression.png)
16. [ConfusionMatirx_DecisionTree](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/ConfusionMatrix_Decision%20Tree.png)
17. [ConfusionMatirx_RandomForest](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/ConfusionMatrix_Random%20Forest.png)
18. [ConfusionMatirx_KNN](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/ConfusionMatrix_KNN.png)
19. [ConfusionMatirx_XgBoost](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/ConfusionMatrix_Xgboost.png)
20. [ConfusionMatirx_SVM_SGD](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/ConfusionMatrix_SVM_SGD.png)
    

## Model Evaluation

<b> Comparision of Models without any params or tuning </b>

<img width="700" alt="" src="https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/ModelsBeforeTuning.png">

<b> Improved Models with params tuning </b>

<img width="700" alt="" src="https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/ModelsAfterTuning.png">

## Comparision of Classification Metrices
<br></b>
<img width="700" alt="" src="https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/ClassificationMetricComparision.png">

## ROC Curve Analysis
<br></b>
<img width="700" alt="" src="https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/images/ROCCurveComparision.png">

## Recommendation and Conclusion

1. The Random Forest model has demonstrated the best overall metrics in this analysis, making it a reliable tool for predicting the churn of customer.
2. XGBoost also have great metrices and is comparable to Random forest.
3. ROC curve also shows that RandomForest and XgBoost are closer to top left corner and having AUC value close to 1. Hence are better classifier.
4. The model performs good and makes correct predictions in most cases.
5. While the model is performing good, there is still a significant rate of false negatives, which means it is not correctly identifying some churns.
6. Using hyperparameter tuning improved the scores.

## Challenges and overcome

1. SVM was not completing processing. Tried with changing the kernel to linear with other options and still didn't help. Then tried using SGD classifier and it worked.
2. Used RandomizedSearch for performance in Hyperparameter tuning as grid search was taking way longer time.
  
## Next Steps

1. Use Neural network to see if we can get better results.
2. Do SHAP analysis to do more interpretation on the models.


## Repository Link (GitHub)

1. [GitHub Link for "Customer Churn Analysis"](https://github.com/CoderNBR/Customer-Churn-Prediction)
2. [Jupyter Notebook](https://github.com/CoderNBR/Customer-Churn-Prediction/blob/main/CustomerChurnPrediction.ipynb)
3. [Images and Vizualization](https://github.com/CoderNBR/Customer-Churn-Prediction/tree/main/images)
3. [Presentation](https://github.com/CoderNBR/Customer-Churn-Prediction/tree/main/Presentation)

