"""
@author: OLimantseva
Titanic Project
"""
import numpy as np
import pandas as pd
from numpy.random import randn
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame 

titanic_df=pd.read_csv('train.csv')
titanic_df.head()
titanic_df.info()

# Distribution of the gender 

sns.factorplot(x='Sex',data=titanic_df,kind='count')
plt.show()

sns.factorplot(x='Sex',data=titanic_df,hue='Pclass',kind='count')
plt.show()

sns.factorplot(x='Pclass',data=titanic_df,hue='Sex',kind='count')
plt.show()

def mfc(passenger):
    age,sex=passenger
    if age<16:
        return 'child'
    else:
        return sex

# Evaluate children from the list of people

titanic_df['person']=titanic_df[['Age','Sex']].apply(mfc,axis=1)

sns.factorplot(x='Pclass',data=titanic_df,kind='count',hue='person')
plt.show()

titanic_df['Age'].hist(bins=70)
titanic_df['Age'].mean() # 29.69911764705882

titanic_df['person'].value_counts()

# Kernel density estimation of the people by age, class, sex

fig=sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

# Distribution, corresponding to the gender and age

fig=sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

# Age difference corresponding to the class

fig=sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

# Evaluation of the people according to the deck

dec=titanic_df['Cabin'].dropna()
levels=[]

for level in dec:
    levels.append(level[0])

levels.sort()
    
cabin_df=DataFrame(levels)
cabin_df.columns=['Cabin']
cabin_df=cabin_df[cabin_df.Cabin != 'T']
sns.factorplot('Cabin',data=cabin_df,palette='winter_d',kind='count')

#

sns.factorplot('Embarked',data=titanic_df,hue='Pclass',row_order=['C','Q','S'],kind='count')

# Who was alone and who was with the family on Titanic.

titanic_df['Alone']=titanic_df.SibSp+titanic_df.Parch

titanic_df['Alone'].loc[titanic_df['Alone']>0]='With family'
titanic_df['Alone'].loc[titanic_df['Alone']==0]='Alone'

sns.factorplot('Alone',data=titanic_df,palette='Blues',kind='count')

# Analysis of the survival rate with respect to class, age and family memebers

titanic_df['Surviver']=titanic_df.Survived.map({0:'No',1:'Yes'})

sns.factorplot('Surviver',data=titanic_df,palette='Blues',kind='count')
sns.factorplot('Pclass','Survived',hue='person',data=titanic_df)
plt.show()
sns.violinplot(x='Pclass',y='Survived',data=titanic_df,inner='stick',orient='v')
plt.show()

sns.lmplot('Age','Survived',data=titanic_df,palette='winter')
plt.show()
generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',data=titanic_df,hue='Pclass',palette='winter',x_bins=generations)
plt.show()
sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)
plt.show()


sns.lmplot('Age','Survived',hue='Alone',data=titanic_df,palette='winter',x_bins=generations)
plt.show()

# Effect of the Deck on the passengers survival rate
    
dec=titanic_df[["Survived","Cabin"]]
dec=dec.dropna()
dec['Deck']=dec['Cabin'].str[:1]
dec=dec[dec.Deck != 'T']
dec=dec.sort_values(by='Deck')
sns.factorplot('Deck',data=dec,palette='winter_d',hue='Survived',kind='count')
