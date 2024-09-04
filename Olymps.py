r/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import seaborn as sns


# In[3]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


from sklearn.linear_model import LinearRegression


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


from sklearn.metrics import mean_squared_error 


# In[8]:


from sklearn.metrics import r2_score


# In[9]:


from sklearn.cluster import KMeans


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


from sklearn.manifold import TSNE


# # Import Data-Frame

# In[13]:


olmpys = pd.read_csv("C:/Users/siddh/Downloads/Olympics/olympic.csv")


# # Identify Shape of Data

# In[61]:


olmpys.shape


# ### Imported data in represented below

# In[15]:


olmpys


# ### Here represents the data of top five Countries in which USA leads

# In[16]:


olmpys.sort_values('Total',ascending =False).head()


# # Data View

# ### This data shows thats how much medals won by a country

# In[17]:


Perform_Country = olmpys[['Country','Total']].sort_values('Total',ascending=False)
Perform_Country


# # Medal Graph

# ### It shows that the decline of the bar chart with respect to the country performance

# In[18]:


Perform_Country.plot(kind = 'bar',width=.75)


# ## Avg Medals

# In[19]:


Total_Medal = olmpys['Total'].sum()
Total_Countries = olmpys['Country'].value_counts().sum()
Avg_Medals = Total_Medal/Total_Countries
Avg_Medals


# In[20]:


int(Avg_Medals)                   


# ## Presentation of India

# ### Indexing and Preformance

# ### This shows the position of India through the indexing methods

# In[21]:


index_of_india = olmpys[olmpys['Country'] == 'India'].index[0]
index_of_india                                                           


# In[22]:


df_India= olmpys.iloc[index_of_india]
df_India


# ## Ranking the Counties

# ### Top 10 Countries

# ### Top Countries come across with there  ranking structure

# In[23]:


olmpys[['Rank','Country']].head(10)


# # Linear Regression

# ### "Linear regression is a type of supervised machine learning algorithm that computes the linear relationship between the dependent variable and one or more independent features by fitting a linear equation to observed data".              The visualization of dataset shows between total medals and ranks
# 
# 
# 
#  

# In[24]:


X = olmpys[['Rank']]
y = olmpys[['Total']]
plt.xlabel('Rank')
plt.ylabel('Total Medals')
plt.scatter(X,y)


# In[25]:


X = olmpys[['Rank']]
y = olmpys[['Total']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=45)


# In[26]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[27]:


C= model.intercept_
C


# In[28]:


m = model.coef_
m


# In[29]:


Y_pred_train = m*X_train + C


# In[30]:


y_pred = model.predict(X_test)


# In[31]:


X_test = X_test.to_numpy().flatten()


# In[32]:


plt.scatter(X_train,y_train)
plt.plot(Y_pred_train,X_train, color ="yellow")
plt.scatter(y_test,y_pred, color='red')
plt.xlabel('Total Medals')
plt.ylabel('Rank')


# In[ ]:





# In[33]:


# red - Medels 
# blue -  Rank


# ## Multiple Regression Function

# ### 'Multiple linear regression (MLR), also known simply as multiple regression, is a statistical technique that uses several explanatory variables to predict the outcome of a response variable. The goal of MLR is to model the linear relationship between the explanatory (independent) variables and response (dependent) variables'.
# 
# ### This dataframe shows that the countries won there respective Medals

# In[35]:


olmpys


# In[34]:


olmpys


# #### fromed new Database after droppeing the Series 'Total' as xNew

# In[37]:


xNew = pd.read_csv("C:Users/siddh/Downloads/Olympics/olympic_.csv")   


# ## xNew=xNew.drop(columns = ['Country','Country Code','Total'])

# In[38]:


xNew=xNew.drop(columns = ['Country','Country Code','Total'])


# In[39]:


yNew = olmpys['Total']
yNew


# In[40]:


X_train,X_test,y_train,y_test = train_test_split(xNew,yNew,test_size=0.4, random_state=42)


# In[41]:


model.fit(X_train, y_train)


# In[42]:


cNew = model.intercept_
cNew


# In[43]:


mNew = model.coef_
mNew


# In[44]:


y_pred_train = model.predict(X_train)


# In[45]:


y_pred_train


# In[46]:


plt.scatter(y_train,y_pred_train)
plt.xlabel('Countries in Region')
plt.ylabel('Total Medals')


# In[47]:


r2_score(y_train,y_pred_train)


# In[48]:


y_pred_test = model.predict(X_test)


# In[49]:


y_pred_test


# In[50]:


plt.scatter(y_test,y_pred_test)
plt.plot(y_test,y_pred_test,color='red')
plt.xlabel('Countries in Region')
plt.ylabel('Total Medals')


# In[51]:


r2_score(y_train,y_pred_train)


# ##  K-means Clustering

# In[52]:


newX = olmpys[['Gold', 'Silver', 'Bronze']]


# In[53]:


kmeans = KMeans(n_clusters=4, random_state=42)
olmpys['Cluster'] = kmeans.fit_predict(newX)


# In[54]:


olmpys['Cluster']


# In[55]:


olmpys['Cluster'].head(20)


# In[56]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='Gold', y='Silver', hue='Cluster', size='Bronze', sizes=(50, 200), data=olmpys, palette='viridis', legend='full')
plt.title('K-means Clustering of Countries Based on Gold, Silver, and Bronze Medals')
plt.xlabel('Gold Medals')
plt.ylabel('Silver Medals')
plt.legend(title='Cluster')
plt.show()


# In[57]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='Bronze', y='Silver', hue='Cluster', size='Gold', sizes=(50, 200), data=olmpys, palette='viridis', legend='full')
plt.title('K-means Clustering of Countries Based on Gold, Silver, and Bronze Medals')
plt.xlabel('Bronze Medals')
plt.ylabel('Silver Medals')
plt.legend(title='Cluster')
plt.show()


# In[59]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='Bronze', y='Gold', hue='Cluster', size='Silver', sizes=(50, 200), data=olmpys, palette='viridis', legend='full')
plt.title('K-means Clustering of Countries Based on Gold, Silver, and Bronze Medals')
plt.xlabel('Bronze Medals')
plt.ylabel('Gold Medals')
plt.legend(title='Cluster')
plt.show()


# In[ ]:




