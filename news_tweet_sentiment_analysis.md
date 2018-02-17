

```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_key_secret, 
                    access_token, 
                    access_token_secret)
consumer_secret = consumer_key_secret

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Search Term
target_users = ("@BBCNews", "@CBSNews", "@CNN","@FoxNews","@nytimes")

# Array to hold sentiment analysis
sentiment_array = []

# Dataframe to hold tweent info
columns = ["Twitter Account","Date","Compound","Positive","Negative","Neutral","Text"]
tweet_df = pd.DataFrame()

num_pages = 5
# Loop through all target users
for num_user,target_user in enumerate(target_users):

    # Variables for holding sentiments and tweet info
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    tweet_array = []


    # Loop through 5 pages of tweets (total 100 tweets)
    for x in range(num_pages):

        # Get all tweets from home feed
        response = api.user_timeline(target_user, page=x)
    
        # Loop through all tweets and print the tweet text
        for num_tweet,tweet in enumerate(response):
            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]

            # Add each value to the appropriate array
            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)
            
            tweet_df.at[num_user*num_pages*20+x*20+num_tweet,'Twitter Account'] = target_user
            tweet_df.at[num_user*num_pages*20+x*20+num_tweet,"Date"] = tweet["created_at"]
            tweet_df.at[num_user*num_pages*20+x*20+num_tweet,"Compound"] = compound
            tweet_df.at[num_user*num_pages*20+x*20+num_tweet,"Positive"] = pos
            tweet_df.at[num_user*num_pages*20+x*20+num_tweet,"Negative"] = neu
            tweet_df.at[num_user*num_pages*20+x*20+num_tweet,"Neutral"] = neg
            tweet_df.at[num_user*num_pages*20+x*20+num_tweet,"Text"] = tweet["text"]

    # Create a dictionary of the Average Sentiments
    sentiment = {'Organization':target_user,
                'Compound':np.mean(compound_list),
                'Positive':np.mean(positive_list),
                'Neutral':np.mean(neutral_list),
                'Negative':np.mean(negative_list),
                'Tweet Count':len(compound_list)}

    # Print the Sentiments
    sentiment_array.append(sentiment)
    print(sentiment)
    print()

tweet_df.to_csv('sentiment_analysis_media_tweets.csv')
tweet_df
```

    {'Organization': '@BBCNews', 'Compound': -0.04462300000000001, 'Positive': 0.059930000000000004, 'Neutral': 0.85407000000000011, 'Negative': 0.085979999999999987, 'Tweet Count': 100}
    
    {'Organization': '@CBSNews', 'Compound': -0.13422100000000001, 'Positive': 0.051990000000000001, 'Neutral': 0.83208000000000015, 'Negative': 0.11591999999999998, 'Tweet Count': 100}
    
    {'Organization': '@CNN', 'Compound': -0.071068000000000006, 'Positive': 0.057229999999999996, 'Neutral': 0.85742999999999991, 'Negative': 0.085329999999999989, 'Tweet Count': 100}
    
    {'Organization': '@FoxNews', 'Compound': -0.027860999999999997, 'Positive': 0.063219999999999998, 'Neutral': 0.86237999999999981, 'Negative': 0.074380000000000002, 'Tweet Count': 100}
    
    {'Organization': '@nytimes', 'Compound': 0.046994000000000001, 'Positive': 0.073080000000000006, 'Neutral': 0.86616999999999988, 'Negative': 0.060749999999999992, 'Tweet Count': 100}
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Twitter Account</th>
      <th>Date</th>
      <th>Compound</th>
      <th>Positive</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 19:03:17 +0000 2017</td>
      <td>-0.4019</td>
      <td>0.000</td>
      <td>0.748</td>
      <td>0.252</td>
      <td>Dalston bus ticket clash: Eight police officer...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 18:53:04 +0000 2017</td>
      <td>0.4767</td>
      <td>0.339</td>
      <td>0.661</td>
      <td>0.000</td>
      <td>Children's commissioner may consider legal act...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 18:01:06 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCSport: Half an hour gone.\n\nArsenal 0-...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 17:53:21 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @bbcweather: Heading out this #SaturdayNigh...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 17:28:39 +0000 2017</td>
      <td>-0.5994</td>
      <td>0.000</td>
      <td>0.804</td>
      <td>0.196</td>
      <td>RT @BBCNewsbeat: The family of 14-year-old Sam...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 17:02:29 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCSport: What a story for Hereford FC - t...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 16:55:38 +0000 2017</td>
      <td>-0.2263</td>
      <td>0.000</td>
      <td>0.899</td>
      <td>0.101</td>
      <td>RT @BBCNewsbeat: Tangled tinsel and wonky tree...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 16:48:36 +0000 2017</td>
      <td>0.8934</td>
      <td>0.416</td>
      <td>0.584</td>
      <td>0.000</td>
      <td>RT @BBCSport: Wales survived a remarkable come...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 15:57:36 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCSport: Here are the half-time Premier L...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 15:29:08 +0000 2017</td>
      <td>-0.2960</td>
      <td>0.000</td>
      <td>0.789</td>
      <td>0.211</td>
      <td>RT @BBCNewsbeat: Motorway Martin: PC single-ha...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 15:02:01 +0000 2017</td>
      <td>0.5106</td>
      <td>0.320</td>
      <td>0.680</td>
      <td>0.000</td>
      <td>Barclays axes free Kaspersky product as a 'pre...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 14:28:13 +0000 2017</td>
      <td>-0.5719</td>
      <td>0.000</td>
      <td>0.684</td>
      <td>0.316</td>
      <td>Whirlpool tumble dryers: MPs' anger as replace...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 14:03:52 +0000 2017</td>
      <td>0.4019</td>
      <td>0.252</td>
      <td>0.748</td>
      <td>0.000</td>
      <td>Chief vet defends support of larger hen cages ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 13:57:30 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCScienceNews: Migraine is 'not just a he...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 12:40:53 +0000 2017</td>
      <td>-0.2960</td>
      <td>0.000</td>
      <td>0.686</td>
      <td>0.314</td>
      <td>Motorway PC stops lorry from falling off bridg...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 12:40:52 +0000 2017</td>
      <td>0.0258</td>
      <td>0.190</td>
      <td>0.657</td>
      <td>0.152</td>
      <td>RT @bbcstories: Errol’s a mechanic and prostat...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 12:23:57 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Boy, 14, dies after being hit on M67 in Hyde h...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 12:03:23 +0000 2017</td>
      <td>0.2023</td>
      <td>0.130</td>
      <td>0.870</td>
      <td>0.000</td>
      <td>Ashes: Australia on top after day one of secon...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 11:54:26 +0000 2017</td>
      <td>0.3400</td>
      <td>0.107</td>
      <td>0.893</td>
      <td>0.000</td>
      <td>RT @5liveSport: That's close of play for Day 1...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 11:40:36 +0000 2017</td>
      <td>-0.4939</td>
      <td>0.000</td>
      <td>0.887</td>
      <td>0.113</td>
      <td>RT @Moneybox: Do we still need cash machines? ...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 19:03:17 +0000 2017</td>
      <td>-0.4019</td>
      <td>0.000</td>
      <td>0.748</td>
      <td>0.252</td>
      <td>Dalston bus ticket clash: Eight police officer...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 18:53:04 +0000 2017</td>
      <td>0.4767</td>
      <td>0.339</td>
      <td>0.661</td>
      <td>0.000</td>
      <td>Children's commissioner may consider legal act...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 18:01:06 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCSport: Half an hour gone.\n\nArsenal 0-...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 17:53:21 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @bbcweather: Heading out this #SaturdayNigh...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 17:28:39 +0000 2017</td>
      <td>-0.5994</td>
      <td>0.000</td>
      <td>0.804</td>
      <td>0.196</td>
      <td>RT @BBCNewsbeat: The family of 14-year-old Sam...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 17:02:29 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCSport: What a story for Hereford FC - t...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 16:55:38 +0000 2017</td>
      <td>-0.2263</td>
      <td>0.000</td>
      <td>0.899</td>
      <td>0.101</td>
      <td>RT @BBCNewsbeat: Tangled tinsel and wonky tree...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 16:48:36 +0000 2017</td>
      <td>0.8934</td>
      <td>0.416</td>
      <td>0.584</td>
      <td>0.000</td>
      <td>RT @BBCSport: Wales survived a remarkable come...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 15:57:36 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCSport: Here are the half-time Premier L...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>@BBCNews</td>
      <td>Sat Dec 02 15:29:08 +0000 2017</td>
      <td>-0.2960</td>
      <td>0.000</td>
      <td>0.789</td>
      <td>0.211</td>
      <td>RT @BBCNewsbeat: Motorway Martin: PC single-ha...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 09:11:02 +0000 2017</td>
      <td>0.7650</td>
      <td>0.292</td>
      <td>0.708</td>
      <td>0.000</td>
      <td>Two Englishmen took a shared love of boat life...</td>
    </tr>
    <tr>
      <th>471</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 08:44:20 +0000 2017</td>
      <td>-0.8555</td>
      <td>0.000</td>
      <td>0.656</td>
      <td>0.344</td>
      <td>A war criminal’s apparent suicide in court lea...</td>
    </tr>
    <tr>
      <th>472</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 08:24:32 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Here are excerpts from Michael Flynn's court d...</td>
    </tr>
    <tr>
      <th>473</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 08:05:54 +0000 2017</td>
      <td>-0.3818</td>
      <td>0.000</td>
      <td>0.794</td>
      <td>0.206</td>
      <td>Mexico’s Government Is Blocking Its Own Anti-C...</td>
    </tr>
    <tr>
      <th>474</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 08:02:21 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Playlist: The Playlist: The Hold Steady Get Re...</td>
    </tr>
    <tr>
      <th>475</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 07:53:35 +0000 2017</td>
      <td>-0.8481</td>
      <td>0.000</td>
      <td>0.620</td>
      <td>0.380</td>
      <td>A bacterium that lives in the mouth is also hi...</td>
    </tr>
    <tr>
      <th>476</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 07:34:37 +0000 2017</td>
      <td>-0.4754</td>
      <td>0.000</td>
      <td>0.838</td>
      <td>0.162</td>
      <td>Why are the 50-year-olds of today more unhappy...</td>
    </tr>
    <tr>
      <th>477</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 07:34:37 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Here's how each senator voted on the Republica...</td>
    </tr>
    <tr>
      <th>478</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 07:17:19 +0000 2017</td>
      <td>0.3400</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.000</td>
      <td>News Analysis: Republicans Near a Big Win — bu...</td>
    </tr>
    <tr>
      <th>479</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 07:00:22 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Editorial: A Historic Tax Heist https://t.co/e...</td>
    </tr>
    <tr>
      <th>480</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 06:56:41 +0000 2017</td>
      <td>0.5859</td>
      <td>0.192</td>
      <td>0.808</td>
      <td>0.000</td>
      <td>The Senate passed its tax overhaul, a major wi...</td>
    </tr>
    <tr>
      <th>481</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 06:45:50 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Seeking Russian Presidency, Socialite Hits the...</td>
    </tr>
    <tr>
      <th>482</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 06:41:50 +0000 2017</td>
      <td>0.5574</td>
      <td>0.247</td>
      <td>0.753</td>
      <td>0.000</td>
      <td>People who can’t stand U2’s earnest, heal-the-...</td>
    </tr>
    <tr>
      <th>483</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 06:22:07 +0000 2017</td>
      <td>-0.5106</td>
      <td>0.000</td>
      <td>0.752</td>
      <td>0.248</td>
      <td>RT @nytpolitics: A Hasty, Hand-Scribbled Tax B...</td>
    </tr>
    <tr>
      <th>484</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 06:06:03 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @thomaskaplan: The Senate is about to vote ...</td>
    </tr>
    <tr>
      <th>485</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 05:52:08 +0000 2017</td>
      <td>0.2023</td>
      <td>0.101</td>
      <td>0.899</td>
      <td>0.000</td>
      <td>Senate is voting on amendments after Republica...</td>
    </tr>
    <tr>
      <th>486</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 05:34:03 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @NYTSports: The Yankees have chosen Aaron B...</td>
    </tr>
    <tr>
      <th>487</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 05:32:05 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>By midafternoon on Friday, Republicans had the...</td>
    </tr>
    <tr>
      <th>488</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 05:19:13 +0000 2017</td>
      <td>0.2263</td>
      <td>0.087</td>
      <td>0.913</td>
      <td>0.000</td>
      <td>RT @nytgraphics: LIVE: Senate passes amendment...</td>
    </tr>
    <tr>
      <th>489</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 05:14:09 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @nytpolitics: Latest from the Senate: Vice ...</td>
    </tr>
    <tr>
      <th>490</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 05:02:06 +0000 2017</td>
      <td>0.1531</td>
      <td>0.168</td>
      <td>0.693</td>
      <td>0.139</td>
      <td>James Comey quoted a Bible passage about justi...</td>
    </tr>
    <tr>
      <th>491</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 04:48:43 +0000 2017</td>
      <td>-0.5106</td>
      <td>0.000</td>
      <td>0.708</td>
      <td>0.292</td>
      <td>A Hasty, Hand-Scribbled Tax Bill Sets Off an O...</td>
    </tr>
    <tr>
      <th>492</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 04:47:06 +0000 2017</td>
      <td>0.1027</td>
      <td>0.105</td>
      <td>0.806</td>
      <td>0.089</td>
      <td>RT @NYTNational: As Roy Moore has sought to un...</td>
    </tr>
    <tr>
      <th>493</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 04:32:07 +0000 2017</td>
      <td>0.7096</td>
      <td>0.312</td>
      <td>0.688</td>
      <td>0.000</td>
      <td>The best TV shows and movies new to Netflix, A...</td>
    </tr>
    <tr>
      <th>494</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 04:25:14 +0000 2017</td>
      <td>0.2023</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.000</td>
      <td>Harvard Agrees to Turn Over Records Amid Discr...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 04:10:55 +0000 2017</td>
      <td>0.5994</td>
      <td>0.246</td>
      <td>0.754</td>
      <td>0.000</td>
      <td>RT @nytgraphics: LIVE: How senators voted on M...</td>
    </tr>
    <tr>
      <th>496</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 04:02:28 +0000 2017</td>
      <td>0.2023</td>
      <td>0.114</td>
      <td>0.886</td>
      <td>0.000</td>
      <td>Republicans are on the verge of passing tax bi...</td>
    </tr>
    <tr>
      <th>497</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 03:47:02 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @nytpolitics: For Trump's team, watching on...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 03:32:07 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>What we know about the connections between Tru...</td>
    </tr>
    <tr>
      <th>499</th>
      <td>@nytimes</td>
      <td>Sat Dec 02 03:17:03 +0000 2017</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Experience the new Seven Wonders of the World,...</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 7 columns</p>
</div>




```python
sentiment_df = pd.DataFrame(sentiment_array)
sentiment_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Organization</th>
      <th>Positive</th>
      <th>Tweet Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.044623</td>
      <td>0.08598</td>
      <td>0.85407</td>
      <td>@BBCNews</td>
      <td>0.05993</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.134221</td>
      <td>0.11592</td>
      <td>0.83208</td>
      <td>@CBSNews</td>
      <td>0.05199</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.071068</td>
      <td>0.08533</td>
      <td>0.85743</td>
      <td>@CNN</td>
      <td>0.05723</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.027861</td>
      <td>0.07438</td>
      <td>0.86238</td>
      <td>@FoxNews</td>
      <td>0.06322</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.046994</td>
      <td>0.06075</td>
      <td>0.86617</td>
      <td>@nytimes</td>
      <td>0.07308</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>




```python
#get lists for each news agency with compound score for each tweet
compound_score = {}
for news_organization in target_users:
    compound_score[news_organization] = tweet_df.loc[tweet_df["Twitter Account"]==news_organization]["Compound"].values
compound_score_df = pd.DataFrame(compound_score)
compound_score_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>@BBCNews</th>
      <th>@CBSNews</th>
      <th>@CNN</th>
      <th>@FoxNews</th>
      <th>@nytimes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.4019</td>
      <td>0.3818</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.5574</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.4767</td>
      <td>0.4939</td>
      <td>-0.6369</td>
      <td>-0.5423</td>
      <td>0.6072</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.4404</td>
      <td>0.0000</td>
      <td>-0.6808</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.5859</td>
      <td>0.8225</td>
      <td>0.4215</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.5994</td>
      <td>0.0000</td>
      <td>-0.5106</td>
      <td>0.7845</td>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0000</td>
      <td>0.2023</td>
      <td>0.0000</td>
      <td>0.3400</td>
      <td>-0.7351</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.2263</td>
      <td>0.5719</td>
      <td>-0.3400</td>
      <td>0.4215</td>
      <td>-0.3182</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.8934</td>
      <td>-0.7351</td>
      <td>-0.2263</td>
      <td>-0.5267</td>
      <td>0.6908</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0000</td>
      <td>0.1779</td>
      <td>0.1280</td>
      <td>0.3612</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.2960</td>
      <td>0.0000</td>
      <td>-0.4215</td>
      <td>0.0000</td>
      <td>0.4019</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.5106</td>
      <td>-0.6124</td>
      <td>0.4019</td>
      <td>-0.6800</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.5719</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.3400</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.4019</td>
      <td>0.2500</td>
      <td>-0.5994</td>
      <td>0.7322</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0000</td>
      <td>0.3612</td>
      <td>0.0000</td>
      <td>0.3400</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.2960</td>
      <td>-0.5859</td>
      <td>0.1280</td>
      <td>0.0000</td>
      <td>0.5859</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0258</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.5574</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0000</td>
      <td>-0.2960</td>
      <td>-0.1027</td>
      <td>-0.3400</td>
      <td>0.2263</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.2023</td>
      <td>-0.6478</td>
      <td>-0.3818</td>
      <td>-0.5267</td>
      <td>0.4215</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.3400</td>
      <td>0.0000</td>
      <td>-0.3597</td>
      <td>0.0000</td>
      <td>0.6249</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-0.4939</td>
      <td>-0.7579</td>
      <td>0.4404</td>
      <td>-0.0516</td>
      <td>-0.2732</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-0.4019</td>
      <td>0.3818</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.5574</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.4767</td>
      <td>0.4939</td>
      <td>-0.6369</td>
      <td>-0.5423</td>
      <td>0.6072</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.4404</td>
      <td>0.0000</td>
      <td>-0.6808</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.5859</td>
      <td>0.8225</td>
      <td>0.4215</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-0.5994</td>
      <td>0.0000</td>
      <td>-0.5106</td>
      <td>0.7845</td>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0000</td>
      <td>0.2023</td>
      <td>0.0000</td>
      <td>0.3400</td>
      <td>-0.7351</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-0.2263</td>
      <td>0.5719</td>
      <td>-0.3400</td>
      <td>0.4215</td>
      <td>-0.3182</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.8934</td>
      <td>-0.7351</td>
      <td>-0.2263</td>
      <td>-0.5267</td>
      <td>0.6908</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.0000</td>
      <td>0.1779</td>
      <td>0.1280</td>
      <td>0.3612</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.2960</td>
      <td>0.0000</td>
      <td>-0.4215</td>
      <td>0.0000</td>
      <td>0.4019</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>-0.3400</td>
      <td>-0.5423</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.7650</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.0000</td>
      <td>0.5859</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.8555</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.2500</td>
      <td>-0.8020</td>
      <td>0.4939</td>
      <td>-0.3400</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.4404</td>
      <td>-0.6369</td>
      <td>0.0772</td>
      <td>0.4404</td>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.0000</td>
      <td>-0.5423</td>
      <td>0.1779</td>
      <td>-0.6808</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>75</th>
      <td>-0.5994</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.8481</td>
    </tr>
    <tr>
      <th>76</th>
      <td>-0.7906</td>
      <td>0.0000</td>
      <td>-0.8885</td>
      <td>-0.4019</td>
      <td>-0.4754</td>
    </tr>
    <tr>
      <th>77</th>
      <td>-0.7351</td>
      <td>-0.7351</td>
      <td>0.0000</td>
      <td>0.3400</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.4588</td>
      <td>0.0000</td>
      <td>-0.6249</td>
      <td>-0.4019</td>
      <td>0.3400</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.0000</td>
      <td>-0.0772</td>
      <td>0.0000</td>
      <td>-0.5859</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.0000</td>
      <td>0.5719</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.5859</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.4019</td>
      <td>-0.5859</td>
      <td>0.4019</td>
      <td>0.5267</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.6486</td>
      <td>0.5574</td>
    </tr>
    <tr>
      <th>83</th>
      <td>-0.8074</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.5106</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.4215</td>
      <td>-0.2960</td>
      <td>0.4019</td>
      <td>-0.4043</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>85</th>
      <td>-0.5267</td>
      <td>-0.3400</td>
      <td>-0.4215</td>
      <td>0.0000</td>
      <td>0.2023</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.0000</td>
      <td>-0.2960</td>
      <td>0.0000</td>
      <td>0.8442</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.3182</td>
      <td>0.6124</td>
      <td>0.0000</td>
      <td>0.4404</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.3182</td>
      <td>0.0000</td>
      <td>0.5859</td>
      <td>-0.4203</td>
      <td>0.2263</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.2263</td>
      <td>-0.4588</td>
      <td>0.0000</td>
      <td>-0.2960</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.0000</td>
      <td>0.4678</td>
      <td>0.0000</td>
      <td>-0.4767</td>
      <td>0.1531</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.0000</td>
      <td>-0.3384</td>
      <td>0.0000</td>
      <td>-0.8834</td>
      <td>-0.5106</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0.0000</td>
      <td>-0.2732</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1027</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.0000</td>
      <td>-0.1531</td>
      <td>0.4939</td>
      <td>0.4019</td>
      <td>0.7096</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.3400</td>
      <td>-0.7351</td>
      <td>0.7574</td>
      <td>-0.2263</td>
      <td>0.2023</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.0000</td>
      <td>-0.5106</td>
      <td>-0.6908</td>
      <td>-0.2960</td>
      <td>0.5994</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.0000</td>
      <td>-0.1531</td>
      <td>0.0258</td>
      <td>0.0000</td>
      <td>0.2023</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.0000</td>
      <td>-0.2960</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.6355</td>
      <td>0.2263</td>
      <td>0.4939</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.0000</td>
      <td>0.5267</td>
      <td>-0.5574</td>
      <td>-0.2732</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>




```python
#create plots
x = np.arange(len(compound_score_df.index))
legend_list = []
fig = plt.figure()
ax = plt.subplot(111)
for news_organization in target_users:
    legend_list.append(ax.scatter(x=x,y=compound_score_df[news_organization],label=news_organization))
plt.legend(legend_list,target_users)

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Number of Tweets Ago')
plt.ylabel('Tweet Polarity')
plt.title('Sentiment Analysis of Media Tweets 12/02/17')
plt.savefig('tweet_polarity_vs_tweet_num.png')
plt.show()
```


![png](output_4_0.png)



```python
x = np.arange(len(sentiment_df.index))
sentiment_df["Compound"].plot(x=x,kind='Bar',grid='on')
plt.xticks(x,list(sentiment_df["Organization"]))
plt.ylabel('Tweet Polarity')
plt.title('Overall Media Sentiment based on Twitter 12/02/12')
plt.savefig('tweet_polarity_average.png')
plt.show()
```


![png](output_5_0.png)


## Observable Trends

1) The New York Times is the only outlet with a positive mean tweet polarity.
2) The mean tweet polarity for CBS News is more than twice as negative as any of the other news outlets.
3) The mean tweet polarity for Fox News has the smallest absolute value and thus is the "least polarized" of the news outlets analyzed.
