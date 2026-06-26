from collections import Counter
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


sns.countplot(data=df, x='sentiment')
plt.show()


#missing values
missing = df.isnull().sum()
print(f"Missing values:\n{missing}")

#sentiment and entity visualization
df["sentiment"].value_counts().plot(kind="bar")
df["entity"].value_counts().head(5).plot(kind="bar")

plt.title("Top 5 Most Mentioned Entities")
plt.xticks(rotation=0)
plt.show()

#kill visualization
no_kill_count = len(df) - kill_count

plt.figure(figsize=(6,5))

plt.bar(
    ["Contains 'kill'", "Does Not Contain 'kill'"],
    [kill_count, no_kill_count]
)

plt.title("Tweets Containing the Word 'Kill'")
plt.xlabel("Category")
plt.ylabel("Number of Tweets")

plt.show()

#lengths of tweets
df["Tweet Length"] = df["text"].str.len()

plt.hist(df["Tweet Length"], bins=30)
plt.title("Distribution of Tweet Length")
plt.xlabel("Characters")
plt.ylabel("Number of Tweets")
plt.show()

#showing average tweet length by sentiment
df.groupby("sentiment")["Tweet Length"].mean().plot(kind="bar")

plt.title("Average Tweet Length by Sentiment")
plt.ylabel("Average Characters")
plt.xticks(rotation=0)
plt.show()

#boxplot of tweet lengths by sentiment
plt.figure(figsize=(10,6))

sns.boxplot(data=df, x="sentiment", y="Tweet Length")

plt.title("Tweet Length Distribution by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Length (Characters)")
plt.xticks(rotation=0)

plt.show()



#most common words in tweets
text = " ".join(df["text"].dropna().astype(str))

text = text.lower()

words = text.split()

# Words to ignore
stop_words = [
    "the", "a", "an", "and", "or", "to", "of", "is", "it",
    "in", "for", "on", "this", "that", "i", "you", "my",
    "me", "we", "our", "your", "be", "are", "was", "with",
    "at", "as", "have", "has", "had", "will", "would",
    "can", "could", "should", "from", "by", "about", "if",
    "but", "they", "them", "their", "he", "she", "his",
    "her", "its", "just", "so", "not", "no", "yes", "all", "any", "some", "more", "most",
    "other", "than", "then", "when", "where", "who", "/", "what", "how", "why", "which", "these", "those", "also", "because", "while", "after", "before", "during", "between", "among", "through", "over", "under", "again", "further", "once",
    "@", ".", "!", "?", ",", ".", ":", ";", "-", "_", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "#", "$", "%", "^", "&", "*", "+", "=", "|",
    "get", "out", "now", "new", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "like", "do", "really", "i'm", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "see", "know", "think", "want", "need", "feel", "say", "make", "go", "come", "take", "give", "look", "use", "find", "tell", "ask", "work", "try",
    "up", "down", "left", "right", "back", "forward", "around", "through", "over", "under", "above", "below", "inside", "outside","been", "it's", "still", "even", "much", "many", "most", "some", "any", "all", "each", "every", "few", "several", "both", "either", "neither", "another", "other", "others", "such", "'","`","got",
    "`","'",

]

# Keep only important words
filtered_words = []

for word in words:
    if word not in stop_words:
        filtered_words.append(word)

word_count = Counter(filtered_words)

top_words = word_count.most_common(10)

word_names = []
word_totals = []

for item in top_words:
    word_names.append(item[0])
    word_totals.append(item[1])

plt.figure(figsize=(10,5))
plt.bar(word_names, word_totals)

plt.title("Top 10 Most Common Words in Tweets")
plt.xlabel("Words")
plt.ylabel("Number of Times Word Appears")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()