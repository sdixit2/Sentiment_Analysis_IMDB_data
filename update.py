#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
import sqlite3
import numpy as np
import os


# In[7]:


from vectorizer import vect
from vectorizer import tokenizer


# In[8]:


# Instead of updating the model everytime , An alternative soultion to the problem will be to download the feedback data going to the 
# sqlite and use it to update the model


# In[11]:


def update_model(db_path, model, batch_size =10000):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * FROM review_db')
    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:,0]
        y = data[:,1].astype(int)
        classes = np.array([0,1])
        X_train = vect.transform(X)
        model.partial_fit(X_train,y,classes = classes)
        results = c.fetchmany(batch_size)
    conn.close()
    return model

clf = pickle.load(open(os.path.join('pk1_objects','classifier.pk1'),'rb'))
db = os.path.join('reviews.sqlite')


# In[12]:


clf = update_model(db_path = db, model = clf, batch_size = 10000)


# In[13]:


# Uncomment the following lines if you are sure that # you want to update your classifier.pkl file # permanently.
# pickle.dump( clf, open( os.path.join(cur_dir,'pkl_objects', 'classifier.pkl'), 'wb'), # protocol = 4) 


# In[ ]:


# we could also fetch one entry at a time by using fetchone instead of fetchmany, which would be computationally very inefficient. However, keep in mind that using the alternative fetchall method could be a problem if we are working with large datasets that exceed the computer or server's memory capacity.


# In[ ]:




