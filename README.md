# bank_customer
Project- Default or Non-default bank customers prediction through Random Froest Classifier

# Credit default prediction -----Flask app-----
Dataset---A dataset from kaggle Credit risk  (link - "https://www.kaggle.com/datasets/laotse/credit-risk-dataset")  

Columns and theie meaning
| Feature Name                  | Description                                  |
|-------------------------------|----------------------------------------------|
| `person_age`                  | Age                                          |
| `person_income`               | Annual Income                                |
| `person_home_ownership`       | Home ownership                               |
| `person_emp_length`           | Employment length (in years)                 |
| `loan_intent`                 | Loan intent                                  |
| `loan_grade`                  | Loan grade                                   |
| `loan_amnt`                   | Loan amount                                  |
| `loan_int_rate`               | Interest rate                                |
| `loan_status`                 | Loan status (0 is non-default, 1 is default) |
| `loan_percent_income`         | Percent income                               |
| `cb_person_default_on_file`   | Historical default                           |
| `cb_person_cred_hist_length`  | Credit history length                        |

here the 'loan_staus' is the column to be predicted and others are the features which will be used for prediction.


File Structure
----------------------

bank_customer/
│
├── app.py                               # Main application file
├── bank-customer-classification.ipynb   # ipbny fle of creating the model
├── credit_risk_dataset.csv.zip          # dataset for training   
├── encoders.pkl                         # encoders for categorical column
├── rfc_model.pkl                        # Model for the prediction
└─── templates/                          # Folder for HTML files
|     ├── home.html                      # Main/home page    
|     └── prediction.html                # Prediction result page
└── README.md                            # Project documentation



Used Random Forest Classifier accuracy score is 92.49%
