#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


salary_df=pd.read_csv('Salary_Data.csv')
salary_df.head()


# In[3]:


salary_df.info()


# In[4]:


salary_df.describe()


# In[9]:


plt.figure(figsize=(6,5))
sns.scatterplot(data=salary_df,x='YearsExperience',y='Salary')
plt.title('Experiance vs salary')


# In[10]:


salary_df.corr()


# In[11]:


sns.distplot(salary_df['Salary'],color='Blue');


# In[12]:


sns.distplot(salary_df.YearsExperience)


# In[16]:


sns.regplot(x='YearsExperience',y='Salary',data=salary_df,color='purple');


# In[17]:


#splitting the datset
X=np.array(salary_df['YearsExperience']).reshape(-1,1)
Y=np.array(salary_df['Salary']).reshape(-1,1)


# In[18]:


Model = LinearRegression()
Model.fit(X,Y)


# In[19]:


predicted=Model.predict(X)


# In[20]:


from sklearn.metrics import mean_absolute_error
MAE=metrics.mean_absolute_error(Y,predicted)
print("Mean absolute error is {}".format(MAE))


# In[21]:


from sklearn.metrics import r2_score
Rsquare= r2_score(Y,predicted)
print("The Rsquare value is {}".format(Rsquare))


# In[22]:


print("The intercept value is {}".format(Model.intercept_))


# In[23]:


print("The slope value is{}".format(Model.coef_))


# In[24]:


import statsmodels.formula.api as smf
model_smf=smf.ols('Salary~YearsExperience',data=salary_df).fit()
print(model_smf.summary())


# In[25]:


model_smf.tvalues,model_smf.pvalues


# In[27]:


new_data=pd.Series([2,5,9,10,12])
pdata=pd.DataFrame(new_data,columns=["YearsExperiance"])
pdata


# In[28]:


Model.predict(pdata)


# In[29]:


x_log=np.log(X)
y_log=np.log(Y)


# In[30]:


model1=LinearRegression()


# In[31]:


model1.fit(x_log,y_log)


# In[32]:


log_pred=model1.predict(x_log)


# In[33]:


log_r2score=r2_score(y_log,log_pred)
print("The rsquare value after transforming the variables into log is {}".format(log_r2score))


# In[34]:


x_sq=X*X
y_sq=Y*Y
model2=LinearRegression()
model2.fit(x_sq,y_sq)


# In[35]:


sq_pred=model2.predict(x_sq)
sq_r2core=r2_score(y_sq,sq_pred)
print("The rsquare value after transforminh the variables into squares is {}".format(sq_r2core))


# In[36]:


x_sqrt=np.sqrt(X)
y_sqrt=np.sqrt(Y)


# In[37]:


model3=LinearRegression()
model3.fit(x_sqrt,y_sqrt)


# In[38]:


sqrt_pred=model3.predict(x_sqrt)


# In[39]:


sqrt_r2core=r2_score(y_sqrt,sqrt_pred)
print("The rsquare value after transforminh the variables into squares is {}".format(sqrt_r2core))


# In[40]:


pd.DataFrame({"models":['model','model(Log)','model(squre)','model(squareroot)'],"rsquare value":[Rsquare,log_r2score,sq_r2core,sqrt_r2core]})


# In[ ]:




