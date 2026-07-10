from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import re


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

df = pd.read_csv('twitter_training.csv', header=None,
                  names=['tweet_id', 'entity', 'sentiment', 'text'])



# print(df.head())
# df.info()
# print(df.describe())
# print(df.isnull().sum())

# print(df['sentiment'].value_counts())

# kill_count = df['text'].str.contains('kill', case=False, na=False).sum() ##"kill" count
# print(f"Number of tweets containing 'kill': {kill_count}")
# print(f"out of: {len(df)} tweets")#total num of tweets


# sns.countplot(data=df, x='sentiment')
# plt.show()


# #missing values
# missing = df.isnull().sum()
# print(f"Missing values:\n{missing}")

# #sentiment and entity visualization
# df["sentiment"].value_counts().plot(kind="bar")
# df["entity"].value_counts().head(5).plot(kind="bar")

# plt.title("Top 5 Most Mentioned Entities")
# plt.xticks(rotation=0)
# plt.show()

# #kill visualization
# no_kill_count = len(df) - kill_count

# plt.figure(figsize=(6,5))

# plt.bar(
#     ["Contains 'kill'", "Does Not Contain 'kill'"],
#     [kill_count, no_kill_count]
# )

# plt.title("Tweets Containing the Word 'Kill'")
# plt.xlabel("Category")
# plt.ylabel("Number of Tweets")

# plt.show()

# #lengths of tweets
# df["Tweet Length"] = df["text"].str.len()

# plt.hist(df["Tweet Length"], bins=30)
# plt.title("Distribution of Tweet Length")
# plt.xlabel("Characters")
# plt.ylabel("Number of Tweets")
# plt.show()

# #showing average tweet length by sentiment
# df.groupby("sentiment")["Tweet Length"].mean().plot(kind="bar")

# plt.title("Average Tweet Length by Sentiment")
# plt.ylabel("Average Characters")
# plt.xticks(rotation=0)
# plt.show()

# #boxplot of tweet lengths by sentiment
# plt.figure(figsize=(10,6))

# sns.boxplot(data=df, x="sentiment", y="Tweet Length")

# plt.title("Tweet Length Distribution by Sentiment")
# plt.xlabel("Sentiment")
# plt.ylabel("Tweet Length (Characters)")
# plt.xticks(rotation=0)

# plt.show()

# # Count how many Positive, Negative, Neutral, and Irrelevant tweets each entity has
# entity_sentiment = df.groupby("entity")["sentiment"].value_counts().unstack(fill_value=0)

# # Convert those counts into percentages
# entity_percent = entity_sentiment.div(entity_sentiment.sum(axis=1), axis=0) * 100

# # Print the top 10 entities for each sentiment percentage
# print("Top 5 Highest Positive Percentage")
# print(entity_percent["Positive"].sort_values(ascending=False).head(5))

# print("\nTop 5 Highest Negative Percentage")
# print(entity_percent["Negative"].sort_values(ascending=False).head(5))

# print("\nTop 5 Highest Neutral Percentage")
# print(entity_percent["Neutral"].sort_values(ascending=False).head(5))

# print("\nTop 5 Highest Irrelevant Percentage")
# print(entity_percent["Irrelevant"].sort_values(ascending=False).head(5))

# # Show the top 5 most mentioned entities
# top_entities = df["entity"].value_counts().head(5).index

# # Graph the sentiment percentages for those entities
# entity_percent.loc[top_entities].plot(kind="bar", figsize=(12,6))

# plt.title("Sentiment Percentage for Top 5 Most Mentioned Entities")
# plt.xlabel("Entity")
# plt.ylabel("Percentage")
# plt.xticks(rotation=45)
# plt.legend(title="Sentiment")

# plt.tight_layout()
# plt.show()

# #most common words in tweets
# #text = " ".join(df["text"].dropna().astype(str))

# positive_text = " ".join(
#     df[df["sentiment"] == "Positive"]["text"].dropna().astype(str)
# )

# negative_text = " ".join(
#     df[df["sentiment"] == "Negative"]["text"].dropna().astype(str)
# )

# #text = text.lower()

# #words = text.split()

# positive_text = re.sub(r"[^a-zA-Z\s]", "", positive_text)
# negative_text = re.sub(r"[^a-zA-Z\s]", "", negative_text)

# positive_text = positive_text.lower()
# positive_words = positive_text.split()

# negative_text = negative_text.lower()
# negative_words = negative_text.split()

# # Words to ignore
# stop_words = [
#     "the", "a", "an", "and", "or", "to", "of", "is", "it",
#     "in", "for", "on", "this", "that", "i", "you", "my",
#     "me", "we", "our", "your", "be", "are", "was", "with",
#     "at", "as", "have", "has", "had", "will", "would",
#     "can", "could", "should", "from", "by", "about", "if",
#     "but", "they", "them", "their", "he", "she", "his",
#     "her", "its", "just", "so", "not", "no", "yes", "all", "any", "some", "more", "most",
#     "other", "than", "then", "when", "where", "who", "/", "what", "how", "why", "which", "these", "those", "also", "because", "while", "after", "before", "during", "between", "among", "through", "over", "under", "again", "further", "once",
#     "@", ".", "!", "?", ",", ".", ":", ";", "-", "_", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "#", "$", "%", "^", "&", "*", "+", "=", "|",
#     "get", "out", "now", "new", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
#     "like", "do", "really", "i'm", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
#     "see", "know", "think", "want", "need", "feel", "say", "make", "go", "come", "take", "give", "look", "use", "find", "tell", "ask", "work", "try",
#     "up", "down", "left", "right", "back", "forward", "around", "through", "over", "under", "above", "below", "inside", "outside","been", "it's", "still", "even", "much", "many", "most", "some", "any", "all", "each", "every", "few", "several", "both", "either", "neither", "another", "other", "others", "such", "'","`","got",
#     "`","'",

# ]

# # Keep only important words
# filtered_positive = []

# for word in positive_words:
#     if word not in stop_words:
#         filtered_positive.append(word)

# positive_count = Counter(filtered_positive)

# top_positive = positive_count.most_common(10)

# filtered_negative = []

# for word in negative_words:
#     if word not in stop_words:
#         filtered_negative.append(word)

# negative_count = Counter(filtered_negative)

# top_negative = negative_count.most_common(10)

# positive_names = []
# positive_totals = []

# for item in top_positive:
#     positive_names.append(item[0])
#     positive_totals.append(item[1])

# negative_names = []
# negative_totals = []

# for item in top_negative:
#     negative_names.append(item[0])
#     negative_totals.append(item[1])



# plt.figure(figsize=(10,5))
# plt.bar(positive_names, positive_totals)

# plt.title("Top 10 Most Common Words in Positive Tweets")
# plt.xlabel("Words")
# plt.ylabel("Number of Times Word Appears")
# plt.xticks(rotation=45)

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,5))
# plt.bar(negative_names, negative_totals)

# plt.title("Top 10 Most Common Words in Negative Tweets")
# plt.xlabel("Words")
# plt.ylabel("Number of Times Word Appears")
# plt.xticks(rotation=45)

# plt.tight_layout()
# plt.show()

# #plt.figure(figsize=(10,5))
# #plt.bar(word_names, word_totals)

# #plt.title("Top 10 Most Common Words in Tweets")
# #plt.xlabel("Words")
# #plt.ylabel("Number of Times Word Appears")

# #plt.xticks(rotation=45)

# #plt.tight_layout()
# #plt.show()


# --------------------------------------------------
# 1. CLEAN THE DATA FOR THE MODEL
# --------------------------------------------------

# Remove rows where text or sentiment is missing
model_df = df.dropna(
    subset=["text", "sentiment"]
).copy()

# Remove duplicate tweets that also have the same sentiment
model_df = model_df.drop_duplicates(
    subset=["text", "sentiment"]
)

print("\nNumber of rows in original dataset:")
print(len(df))

print("\nNumber of rows after model cleaning:")
print(len(model_df))

print("\nSentiment counts after cleaning:")
print(model_df["sentiment"].value_counts())


# --------------------------------------------------
# 2. CREATE THE FEATURES AND LABELS
# --------------------------------------------------

# X is the feature the model uses
# In this project, X contains the tweet text
X = model_df["text"]

# y is the correct answer that the model learns to predict
# In this project, y contains the tweet sentiment
y = model_df["sentiment"]


# --------------------------------------------------
# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nNumber of training tweets:")
print(len(X_train))

print("\nNumber of testing tweets:")
print(len(X_test))


# --------------------------------------------------
# 4. CONVERT TWEET TEXT INTO NUMERICAL FEATURES
# --------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

# Fit learns the vocabulary from the training tweets
# Transform converts the training tweets into numbers
X_train_tfidf = tfidf.fit_transform(X_train)

# Only transform the testing tweets
# This uses the vocabulary learned from the training data
X_test_tfidf = tfidf.transform(X_test)

print("\nTraining feature matrix:")
print(X_train_tfidf.shape)

print("\nTesting feature matrix:")
print(X_test_tfidf.shape)


# --------------------------------------------------
# 5. CREATE AND TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# --------------------------------------------------
# 6. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(
    X_test_tfidf
)


# --------------------------------------------------
# 7. CALCULATE ACCURACY
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# --------------------------------------------------
# 8. PRINT THE CLASSIFICATION REPORT
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 9. DISPLAY THE CONFUSION MATRIX
# --------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation=45
)

plt.title("Tweet Sentiment Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 10. TEST THE MODEL WITH SAMPLE TWEETS
# --------------------------------------------------

sample_tweets = [
    "I absolutely love this game!",
    "This is the worst update ever.",
    "The game update was released today.",
    "Visit the website for more information."
]

# Convert sample tweets using the existing TF-IDF vocabulary
sample_tweets_tfidf = tfidf.transform(sample_tweets)

# Predict the sentiment of each sample tweet
sample_predictions = model.predict(
    sample_tweets_tfidf
)

print("\nSample Tweet Predictions:")

for tweet, prediction in zip(
    sample_tweets,
    sample_predictions
):
    print(f"\nTweet: {tweet}")
    print(f"Predicted sentiment: {prediction}")