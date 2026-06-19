import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('twitter_training.csv', header=None,
                  names=['tweet_id', 'entity', 'sentiment', 'text'])



print(df.head())
df.info()
print(df.describe())
print(df.isnull().sum())

print(df['sentiment'].value_counts())

kill_count = df['text'].str.contains('kill', case=False, na=False).sum() ##"kill" count
print(f"Number of tweets containing 'kill': {kill_count}")
print(f"out of: {len(df)} tweets")#total num of tweets


df['sentiment'].value_counts().plot(kind='bar')
##df['text'].value_counts().plot(kind='pie') trying to display kill count in a pie chart
sns.countplot(data=df, x='sentiment')
plt.show()
##plt.show()