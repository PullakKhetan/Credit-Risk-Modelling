#loading training and testing datasets
#importing common evaluation packages and plotting library
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
traning_data=pd.read_excel("Training Data.xlsx",skiprows=2,index_col="id")  #training data is loaded
test_data=pd.read_excel("Testing Data.xlsx",skiprows=3,index_col="id")    #testing data is loaded
training_output=traning_data['Loan_Status_Flag'] 
test_output=test_data['Loan_Status_Flag']
#creating dummy variable for categorical attributes
traning_data['home_ownership']=traning_data['home_ownership'].astype('category').cat.codes 
test_data['home_ownership']=test_data['home_ownership'].astype('category').cat.codes
traning_data['purpose']=traning_data['purpose'].astype('category').cat.codes
test_data['purpose']=test_data['purpose'].astype('category').cat.codes
#dropping unnecessary attributes from the dataset
train_final=traning_data.drop(['Loan_Status_Flag','verification_status','dti','revol_bal','loan_amnt','int_rate','revol_util','open_acc','annual_inc','pub_rec','total_rec_late_fee'],axis=1)
test_final=test_data.drop(['Loan_Status_Flag','verification_status','dti','revol_bal','loan_amnt','int_rate','revol_util','open_acc','annual_inc','pub_rec','total_rec_late_fee'],axis=1)

##training Decision tree model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(train_final, training_output)
output_pred=classifier.predict(test_final)      #decision tree classifier is created
print(confusion_matrix(test_output, output_pred))
print(classification_report(test_output, output_pred))
f_pos_dect,t_pos_dect,_=roc_curve(test_output,classifier.predict_proba(test_final)[:,1])
plt.figure()
plt.plot(f_pos_dect,t_pos_dect,color='blue',label='Decision Tree')
plt.xlabel('false positive rate',color='white')
plt.ylabel('true positive rate',color='white')
plt.title('ROC Curve',color='white')
plt.legend(loc='best')
plt.show()

##training logistic regression model
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(max_iter=10000)
log_reg.fit(train_final,training_output)
log_reg_pred=log_reg.predict(test_final)   #logistic regression model is created
print(confusion_matrix(test_output,log_reg_pred))
print(classification_report(test_output,log_reg_pred))
f_pos_reg,t_pos_reg,_=roc_curve(test_output,log_reg.predict_proba(test_final)[:,1])
plt.plot(f_pos_reg,t_pos_reg,color='darkorange',label='Logistic Regression')
plt.title('ROC Curve',color='black')
plt.legend(loc='best')
plt.show()

##Training Gradient boosting classifier model
from sklearn.ensemble import GradientBoostingClassifier
gbm_1 = GradientBoostingClassifier(n_estimators = 200, max_depth = 10)
gbm_1.fit(train_final,training_output)
gbm1_yPred = gbm_1.predict(test_final)    #gradient bossting cladssifier is created
print(confusion_matrix(test_output,gbm1_yPred))
print(classification_report(test_output, gbm1_yPred))

##training random forest classifier model
from sklearn.ensemble import RandomForestClassifier
rnd_forest=RandomForestClassifier(n_estimators=200,max_depth=10)
rnd_forest.fit(train_final,training_output)
rnd_forest_pred=rnd_forest.predict(test_final)    #random forest classifier is created
print(confusion_matrix(test_output,rnd_forest_pred))
print(classification_report(test_output, rnd_forest_pred))

##plotting ROC curve for each model
plt.figure()
f_pos_reg,t_pos_reg,_=roc_curve(test_output,log_reg.predict_proba(test_final)[:,1])
f_pos_dect,t_pos_dect,_=roc_curve(test_output,classifier.predict_proba(test_final)[:,1])
f_pos_rndf,t_pos_rndf,_=roc_curve(test_output,rnd_forest.predict_proba(test_final)[:,1])
f_pos_grdb,t_pos_grdb,_=roc_curve(test_output,gbm_1.predict_proba(test_final)[:,1])
plt.plot([0,1],[0,1],'k--')
plt.plot(f_pos_reg,t_pos_reg,color='darkorange',label='Logistic Regression')
plt.plot(f_pos_dect,t_pos_dect,color='blue',label='Decision Tree')
plt.plot(f_pos_rndf,t_pos_rndf,color='green',label='Random Forest')
plt.plot(f_pos_grdb,t_pos_grdb,color='red',label='Gradient Boosting')
plt.xlabel('false positive rate',color='white')
plt.ylabel('true positive rate',color='white')
plt.title('ROC Curve',color='white')
plt.legend(loc='best')
plt.show()

