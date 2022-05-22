#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as mglearn
from sklearn import metrics
import statsmodels.api as sm 


# In[35]:


delivery_df=pd.read_csv('delivery_time.csv')
delivery_df.head()


# In[37]:


delivery_df.info()


# In[40]:


delivery_df.describe()


# In[46]:


delivery_df.isna().sum()


# In[51]:


sns.set(style='white')


# In[52]:


plt.figure(figsize=(6,5))
sns.scatterplot(data=delivery_df,x='Sorting Time',y='Delivery Time')
plt.title('Sorting time vs Delivery time');


# In[55]:


sns.distplot(delivery_df['Delivery Time'],color='lightblue')
plt.title('Distribution of Delivery time');


# In[57]:


sns.distplot(delivery_df['Sorting Time'],color='Grey')
plt.title("Distribution of Sorting time");


# In[58]:


delivery_df.corr()


# In[59]:


sns.regplot(x='Sorting Time',y='Delivery Time',data=delivery_df,color='brown');
plt.title("Regression line")


# In[60]:


X= np.array(delivery_df['Sorting Time']).reshape(-1, 1)
Y= np.array(delivery_df['Delivery Time']).reshape(-1, 1)


# In[64]:


model= LinearRegression()


# In[65]:


model.fit(X,Y)


# In[66]:


predicted=model.predict(X)


# In[67]:


from sklearn.metrics import mean_absolute_error
MAE=metrics.mean_absolute_error(Y,predicted)
print("Mean absolute error is {}".format(MAE))


# In[68]:


from sklearn.metrics import r2_score
Rsquare= r2_score(Y,predicted)
Rsquare


# In[69]:


delivery_df.rename({"Delivery Time":"Delivery_time","Sorting Time":"Sorting_time"},axis=1,inplace=True)
delivery_df.head()


# In[70]:


import statsmodels.formula.api as smf
model1=smf.ols('Delivery_time~Sorting_time',data=delivery_df).fit()
print(model1.summary())


# In[71]:


print("The intercept value is 6.5827\nThe slope is 1.6490 \nThe R-squared value is 0.682\nThe adusted R-squared values is 0.666")


# In[72]:


model1.tvalues,model1.pvalues


# In[73]:


new_data=pd.Series([6,10,8])
dpred=pd.DataFrame(new_data,columns=['Sorting time'])
dpred


# In[74]:


model.predict(dpred)


# In[75]:


delivery_df.head()


# In[76]:


model2 = smf.ols('Delivery_time~np.log(Sorting_time)',data=delivery_df).fit()


# In[77]:


model2.params


# In[78]:


model2.summary()


# In[79]:


print(model2.conf_int(0.01))


# In[80]:


pred_on_model2 = model2.predict(pd.DataFrame(delivery_df['Sorting_time']))


# In[81]:


pred_on_model2.corr(delivery_df['Sorting_time'])


# In[83]:


plt.scatter(x=delivery_df['Sorting_time'],y=delivery_df['Delivery_time'],color='yellow');
plt.scatter(x=delivery_df['Sorting_time'],y=pred_on_model2,color='green');


# In[84]:


model3 = smf.ols('np.log(Delivery_time)~Sorting_time',data=delivery_df).fit()
model3.params


# In[85]:


model3.summary()


# In[86]:


print(model3.conf_int(0.05))


# In[87]:


log_pred=model3.predict(delivery_df.iloc[:,1])
pred_on_model3=np.exp(log_pred)


# In[88]:


plt.subplot(1,2,1)
sns.scatterplot(x=delivery_df['Sorting_time'],y=delivery_df['Delivery_time'])
plt.subplot(1,2,2)
sns.scatterplot(x=delivery_df['Sorting_time'],y=pred_on_model3)


# In[89]:


plt.title("Predicted Vs Actual")
plt.scatter(x=pred_on_model3,y=delivery_df['Delivery_time'],color='purple');plt.xlabel("Predicted");plt.ylabel("Actual")


# In[90]:


delivery_df['Sorting_time_sq']=delivery_df.Sorting_time*delivery_df.Sorting_time
model4=smf.ols('Delivery_time~Sorting_time_sq',data=delivery_df).fit()
model4.params


# In[91]:


model4.summary()


# In[92]:


delivery_df['Sorting_time_sqrt']=np.sqrt(delivery_df['Sorting_time'])


# In[93]:


delivery_df.head()


# In[94]:


model5=smf.ols('Delivery_time~Sorting_time_sqrt',data=delivery_df).fit()
model5.params


# In[95]:


model5.summary()


# In[96]:


pred_on_model5=model5.predict(delivery_df['Sorting_time_sqrt'])


# In[97]:


pred_on_model5.corr(delivery_df['Delivery_time'])


# In[99]:


plt.title("Predicted Vs Actual")
plt.scatter(x=pred_on_model5,y=delivery_df['Delivery_time'],color='blue');plt.xlabel("Predicted");plt.ylabel("Actual")


# In[100]:


pd.DataFrame({"Models":['model1','model2','model3','model4','model5'],"intercept":[model1.params[0],model2.params[0],model3.params[0],model4.params[0],model5.params[0]],"slope":[model1.params[1],model2.params[1],model3.params[1],model4.params[1],model5.params[1]]})


# In[ ]:




