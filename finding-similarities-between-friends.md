# finding similarities between friends by way of social media using shallow and deep natural language processing

<br/>

# abstract

With the surge of social media, the internet has become an interesting and spirited domain in which billions of individuals from all around the world interact, share, post and conduct many daily activities. The immense size of social media data makes it notably different from classic data sources, and the mainly user generated data can be unbelievably noisy and unstructured. In social media mining, social media is considered a world of social atoms (i.e. *individuals*), and entities (e.g. *content, sites, networks etc.*) signaling interactions between social atoms and entities (et al. *Zafarani, Abbasi and Liu*). In this analysis I propose a way of looking at interactions governed solely between social atoms by collecting, mining and measuring the interactions between these social atoms to discover whatever salient patterns reside in each atom as they are projected through the lens of social media.

It is my goal to make sense of a Facebook Messenger chat group belonging to a group of friends that has been active for about 8 years and I hope to find relationships within each members text to derive an overall view of what those relationships mean to each other. The shared meaning within these conversations go back pretty far, and should be reinforced when you look over the entire body of conversational sentences. I do not use the word *meaning* as a way to represent strings of words relating to the intent of a speaker per say. The meaning I'm after in this analysis is derived from *form* rather than context of use. This is to say that we will learn which words are similar in place of commonsense reasoning without truly knowing what each atoms' private mental state infers. 

The overall text fragments within this corpus will be very short, considering the conversations held within the medium are stream of conscious by nature. I will explain in great detail, methods of deriving meaning from these short series of sentences using dense, distributed word embeddings by way of deep learning. After the text has been cleaned, each word will be mapped onto points in a high dimensional space to further reduce *meaningless* words to obtain *meaningful* words for each user. During the text extraction about 25,251 instances from the sum of each users name--originally ending at the 302,733rd instance--was lost. In an attempt to label this wealth of data so that it can be included in modeling user sentiment and topics, I propose a binary classification task to classify each users name to their respective text using an attention-based bidirectional long, short term memory recurrent neural network to learn the relevant features of each users text. The AB-BiLSTMRNN results will be weighted against engineered features tested on more classical information retrieval and shallow learning techniques such as term frequency inverse document frequency and sparse count vectorization trained with semantically weighted word vectors, logistic regression, gradient boosted trees, naive bayes and support vector machines.

<br/>

# introduction

The overall objective is to be able to convert questions about the text...

*What topics are present in the text?*

*Are certain topics shared among all users or just a subset of users?*

*What can we say about the relationship between users and their topics?*

*What will each users separate set of topics say about their association to other users?*

...that are beyond a humans ability to process, into something that can be easily digested. A possible solution for this problem is machine learning, which is a sub-field of artificial intelligence. The overall goal is to teach computers to learn from their own experiences without being strictly designed by rules a human programmed. Think about it like this, machine/deep learning is the computer science equivalent to the realization that the Earth is not the center of the universe; as humans are not the center of intelligence.

Natural language processing is a sub-field of artificial intelligence...

<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/29679899/101290745-2f3b3000-37d2-11eb-813f-8cf472ba58dd.PNG" width="500px">
</p>

<br/>

...that brings together mathematics, computer science and  linguistics to make human language attainable for computers. Natural language processing shares ideas with computational linguistics, but CL is in service of linguistics... 

<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/29679899/101290912-ff405c80-37d2-11eb-91e0-071835432ba9.PNG" width="500px">
</p>

<br/>

...while NLP is centered around the design and exploration of different representations for parsing natural language.  

In Katzian semantics--a form of generativist structural semantics--its said that word meanings are defined in terms of their combination of simpler conceptual components, therefore word meanings are structured entities whose semantic markers reproduce the structure of the represented meaning and whose labels are the words conceptual components. Katzian semantics has its flaws[<a href="https://plato.stanford.edu/entries/word-meaning/#GenSem" title="Katzian semantics" rel="nofollow">'</a>], but the idea that words and their meanings can belong to a structure that lends itself to analysis is still very relevant.  

In an attempt to illustrate the idea of meaning, let's put the words `car`, `bus`, `road` and `driving` into a bag, mix them all up and dump them on the floor. Looking at the words, what do we see? The meaning between a group of words is equal to the distance between them, based on the likeness of their meaning[1]. I'm not just talking about words that appear similar to each other as opposed to just similarity--which can be estimated based on a set of rules, principles and processes that govern the structure of sentences in a given language--what I'm saying is that no matter what order the words are in, `car` will be similar to `bus` and both words are related to `road` and `driving`. This specific type of similarity is the building block of various mathematical tools that are used to estimate the strength of the semantic relationship between each unit of language.

Let's say you want to look at the way a person writes. You could very well end up asking questions like, *"What words are they using? How often are they using similar words?"*. Term frequency-inverse document frequency can be described as a method that looks at a persons vocabulary. Term frequency is how often you use particular words and inverse document frequency is how rare those words are across the document as a whole. The idea is that as people use certain words that aren't common, those are the words that are particularly strong signatures in that individuals writing style. So we'rebasically able to analyze that individuals' vocabulary. 

This method is far from perfect and it ignores some of the most fundamental ideas of meaning. If you're just counting the frequency of words, we're missing out on why those words connect, which is pretty crude. The frequency of a word is somewhat misleading because you have to take into account how words appear together to form something meaningful. When we think of natural language processing as a way to potentially overcome this, we must also make computational linguistic goals. I've taken a sample sentence from one users corpus and used it as input to a linguistics parser built by the NLP Group at Stanford. This program is able to go through the sentence and establish a words syntactic role.

*"Every other perception of soccer in the US is sitting through a grueling game of 9 years olds running in circles."*

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/101291038-f00dde80-37d3-11eb-8532-cdf158534cac.png" width="500px">
</p>

<br/>

This parse tree was trained by a language model, and its looking for various parts of speech within the sentence by looking over all noun and verb phrases. 

A core aspect of NLP from a machine learning/statistical perspective are language models. Language models help machines learn that certain words and sentences are more probable than other words and sentences. So a LM could infer that *My pots and pans are in the dishwasher* is more probable than *My pots and pans are in the school bus*. 

To simplify how the LM behind this dependency parser focuses on the relationships between the words in the *soccer* sentence, take a look at the part of speech (pos) tags for each word and you can see what role each word plays within the sentence...

<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/29679899/101291138-94902080-37d4-11eb-82a6-6b13efc3bb51.PNG" width="500px">
</p>


<br/>

DT = Determiners are words placed in front of a noun to make it clear what the noun refers to. 

JJ = Adjectives are used to describe people, places and things.

NN = Generally, nouns come with many rules and exceptions. Nouns can be gendered, singular, plural, countable, uncountable, definite pronouns, indefinite pronouns, compound nouns and also possessive. There are also capitalization rules which dictate relations to nationalities/ethnicities.

IN = Preposition/subordinating conjunctions. Words that are sometimes conjunctions can act as prepositions when they are followed by objects rather than dependent clauses. For example, the preposition 'of' refers to the belonging to, relating to or connecting with the 'perception' of a group of people. 

NN = Singular nouns in this case names one person, place, thing or idea in all instances in the sentence. 

NNP = Singular proper nouns are more specific than just singular nouns. It gives us the actual name of a person, place, thing or idea. 

VBZ = A third person singular present tense irregular verb. It's commonly used for main verbs and auxiliary verbs in the simple present tense.

VBG = A present tense participle or a word ending in -ing. It represents a form that is derived from a verb but functions as a noun.

NNS = A plural form of common nouns that can be divided into countable nouns of two larger groups: regular and irregular. The usage in this sentence is regular. 

Generally, linguistics is centered around how sentences are basic units of thought, so we can conclude that if things start appearing together in a sentence a lot, we can take away meaningful structures from sentences. The parsed words in the sentence above belong to a corpus of extremely unstructured data, and while these are relatively unstructured data points, at the sentence and word level they are structured. This is exciting because we can begin to look at text as being loosely structured sequences from which we can extract meaning. 

<br/>

# 1. collecting, and extracting the data

The text that I will be using came from Facebook.

<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/29679899/101291371-3feda500-37d6-11eb-900d-e9fc26c08faa.PNG" width="500px">
</p>

<br/>

After Facebook authorized the download, I was able to go through most of the data I've put on their platform over the years, but the only thing I'm interested in is the wealth of texts from Facebook Messenger that I've accumulated in this group:

<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/29679899/101291409-82af7d00-37d6-11eb-95b3-d58f9fbde990.PNG" width="200px">
</p>

<br/>

All of the messages for my intended corpus are now accessible through a nice little (98.4MB) html file. As I scroll through this html file in the browser, I notice a few things. Each users response to the messenger app contains four pieces of data. Their names (which will remain blurred), message and the date/time in which they made a response on the app. Simple enough: 

<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/29679899/101291432-aa064a00-37d6-11eb-9dc3-1283d94e85bc.PNG" width="500px">
</p>

<br/>

Web pages are nice when you're browsing the internet, but analyzing the text would be a lot easier if I could aggregate each persons message into a row that corresponds with their name, date and time of their response. Understanding the logic of the html structure is necessary before any information can be extracted, so if we view the html page's source we can get a closer look.

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101291471-02d5e280-37d7-11eb-934b-bb7e9c56d99b.gif" width="500px">
</p>

<br/>

In the gif above all the information we need is nested within div tags. A div tag is simply a container that encloses page elements in an html file and divides the html file into sections. What we're interested in is what's inside the div containers. Specifically the user names, which are in the `"_3-96 _2pio _2lek _2lel"` div class, their messages are in the `"_3-96 _2let"` div class and the date & time is in the `"_3-94 _2lem"` div class. The div classes--which is where your distinctive markers will usually reside--within the nested div tags contain unique identifiers that will allow us to extract all the div containers that have class attribute `"_3-96 _2pio _2lek _2lel"`, `"_3-96 _2let"` or `"_3-94 _2lem"`. We can scrape each users text, dates and times from this web page, because each piece of information is nested within a particular variable. To scrape the text, we will use a common parsing module called BeautifulSoup.

```python
# dependencies

import warnings
warnings.filterwarnings('ignore')
import bs4
import csv
import urllib
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import Request
```

In the figure above, the `bs4` element will represent a `BeautifulSoup` variable, which will represent the entire html document. More importantly, its a tag object that represents any number of methods that will allow you to replace strings with other strings, define methods for navigating and looking through parse trees etc. 

`urllib.request` is an element that defines a set of functions and classes that will open the URL location of the html file. More specifically, the `Request` method allows vanilla HTTP requests to execute, alleviating the need to encode parameters (it has a built in JSON decoder) and it simply takes a dictionary as an argument. This method will make importing the raw html file into the script very easy. Python will then read the contents of the `bt_message_data` variable as a large string that `BeautifulSoup` will process and the `csv` element will export the processed text into a tabular format.

```python
bt_message_data = urllib.request.urlopen('file:///Users/christianth/Downloads/facebook-christianhardy1023%20(1)/messages/bluestraveler2000_6dec36cd06/message.html', timeout=1)

bt = bt_message_data.read()
soup = bs4.BeautifulSoup(bt, 'html.parser') 
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/29679899/101291640-509f1a80-37d8-11eb-927f-3f6207fd270b.PNG" width="450px">
</p>

Now that the scraping environment is set up, we can begin the process of extracting the information. The `csv.writer()` module will convert the user text into delimited strings that will be stored inside a CSV file. In order to analyze the text further it needs to be in a format that the model can understand, which means we'll need to store the user names, messages and date/time text in a table, so the `writer()` function will create an object suitable for writing the data to our file. To iterate the text over the rows of the CSV file, we need to place the strings `Name`, `Date` & `Time` and `Message` inside the `writerow()` function as arguments, which will give each respective feature it's own label and will represent the column for each features instance in the CSV file. 

Earlier I mentioned how `BeautifulSoup` can define methods for searching through parse trees for specific pieces of data. The `data_name`, `data_date_time` and `data_message` variables contain the soup which is appended to the `find_all()` method. By passing the specified div classes as arguments to `find_all()`, it is now possible to extract the `name`, `message` and `date/time` features from the html soup. Calling the `writerow()` function within a for loop will iterate each observation for a given user over the entirety of the html document and simultaneously write each observation to the CSV file created by `csv.writer()`.

```python
# names of users 

d = csv.writer(open('bt_name_data_R.csv', 'w'))
d.writerow('Name')
data_name = soup.find_all('div', class_ = '_3-96 _2pio _2lek _2lel')
for data_name in data_name:
    names = data_name.contents
    d.writerow(names)
    
# users dates & times 

d = csv.writer(open('bt_date_data_R.csv', 'w'))
d.writerow(['Date & Time'])
data_date_time = soup.find_all('div', class_ = '_3-94 _2lem')
for data_date_time in data_date_time:
    dates_times = data_date_time.contents
    d.writerow(dates_times)
    
# users messages 

d = csv.writer(open('bt_message_data.csv', 'w'))
d.writerow(['Message'])
data_message = soup.find_all('div', class_ = '_3-96 _2let')
for data_message in data_message:
    messages = data_message.get('_3-96 _2let')
    d.writerow([messages])
```
<br/>

# 2. initial data exploration 

Datasets are like a good satirical bildungsroman[<a href="https://en.wikipedia.org/wiki/Bildungsroman" title="What is a bildungsroman" rel="nofollow">'</a>]. They will give you many ambiguous ideas, and sometimes they will even have an interesting story to tell. Exploring your data is a very important pillar of data analysis, because it gives a sense of what can be done with it and what may be possible. The first few reduction methods in this post will attempt to deconstruct the signal in our dataset into a set of features that will help our algorithms learn something meaningful. The only way you can handcraft good features for your data is by visualizing it, and after extracting the text from the html file and exporting everything into a nice tabular format, our first visualization is just a matter of several dependencies and less than 10 lines of code.

```python
# dependencies

import string
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
import re
stopwords = stopwords.words('english')
color = sns.color_palette()
```

`pandas` will allow the text to be read into python, and allow easy handling of various data structures, as well as providing a few other tools for data manipulation. `string` is a built-in python class that gives us the ability to do complex variable substitutions and value formatting.  `matplotlib` is a popular visualization tool that's built on the `numpy` and `scipy` frameworks and is capable of simple 2D plots with limited 3D support. We'll also use `seaborn`, a visualization library based on `matplotlib`. For our use case it explicitly supports relational, and categorical related visualizations as well as an API that supports distributions.  

The `collections` module can implement unique container datatypes that give us options over python's more inclusive containers such as the `dict`, `list`, `set` and `tuple`.  `nltk`, which is widely used in research and industry, is a very important library that will be used throughout the analysis for its suite of text preprocessing libraries for tokenization, stemming, lemmatization, tagging, parsing and semantic reasoning. `re` will handle regex commands and the `color` object carries an important function for generating discrete color palettes which will give our sns plots a bit of color variation.

During this initial exploration, we'll want to isolate and remove a significant portion of frequently used common words that have little contextual meaning on their own. To do this we'll need the `stopwords` object. Stop words can include words like `'the, and, in, of, to'` etc. These frequent, grammatical filler words play an important role in grammar. For instance, when predicting which word should come next in a sequence of words for a natural language generation or understanding task, but this part of the analysis is not complex enough to warrant that level of minuet understanding. 

```python
# data

dataset = pd.read_csv('bt_fb_messenger_data.csv').fillna(' ')

# shape of data
print("Training Data Shape : ", dataset.shape)
```

`pd.read_csv` is a function in `pandas` that allows data to pass into python, but unlike python's native `.read()` function, we can perform containerization operations that are exclusive and specific to the pandas library. `pd.read_csv` is contained within the dataset variable which is then used to determine the shape of the data using the .shape function. After executing these two lines of code, a `pandas` matrix is returned. A matrix in pandas is synonymous with arrays in python. Arrays are data structures that can hold any rendition of data in a structured way, and a matrix is simply a two-dimensional data structure where numbers are arranged into rows and columns. The returned integer tells us that there are 302,731 rows, within the table and 4 columns in a 2-dimensional state. 

In order to understand the text, we'll first need to get a sense for how the variables are distributed. Distributions show us the possible values for variables and how often they occur. Maybe we want to compare the distribution of a variable across levels of other variables. Are we working with univariate, bivariate or multivariate distributions? You can easily make some of these assumptions by just looking at the data in its raw form, but visualizing it helps us to understand it on a more intuitive level. 

Since each users response in the app is dictated by the amount of times they send a message, lets obtain the frequency of each users response. We'll isolate the feature `Names` from the dataset in the names variable to determinehow many total occurrences there are for each user. To visualize the frequency, we'll specify the plots height and width with the figsize element and call it from the `plt.figure` function.

`sns.barplot` supports the input data from `.index`, which specifies where the index values begin, and we use this to access the `pandas` data frame within the names variable, and `.values` allows the index values to be displayed. alpha specifies a semi-translucent tone to the bar plot and color simply returns a color for the bars. `plt.xlabel` and plt.ylabel applies a label to the `x/y` axis and `plt.show()` allows us to visualize the 6 lines of code.

```python
# number of occurances for each user

names = dataset['Name'].value_counts()
plt.figure(figsize=(12,4)) # 36,16 # 12,4
sns.barplot(names.index, names.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('User Name', fontsize=12)
plt.show()
```
<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101365465-d4f1ac00-3871-11eb-8d89-fc8f2a8e3155.png" width="700px">
</p>

<br/>

In the order from greatest to least, `bt_1` has the most frequent occurrences, `bt_2` has the second most frequent occurrences, `bt_3` and `bt_4` have about the same as each other, etc.

<br/>

```python
bt_1 = 80,000

bt_2 = 60,000

bt_3 = 39,000

bt_4 = 39,000

bt_5 = 24,000

bt_6 = 14,000 

bt_7 = 12,000 

bt_8 = 12,000 

bt_9 =  9,000 

bt_10 = 8,000 

bt_11 = 1,200 

bt_12 = 1,100 

bt_13 = 12
```

It's pretty obvious that `bt_13` is an outlier and should be discarded. It's hard to say whether `bt_11` and `bt_12` should be treated as outliers given their disproportionate size. The weight of their contribution will be considered as we go on.

Notice how the tail decay in the plot above starts to fatten and get heavier around the third user followed by a continuum of decreasing values. In order to understand why heavy tailed distributions matter and what it means for our dataset, we'll need to explore a few statistical laws and their similarities.  

<br/>

# power laws

We can define PLs as a relationship between two things. A respective change in one thing results in a proportional respective change in the other thing, and they're both independent of size. More simply, they imply that a small number of events, or in our case words for each user are frequent (common words), while a large number of words are infrequent (rare words).

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101366130-93adcc00-3872-11eb-9cf1-31fc223640ce.png" width="500px">
</p>

<br/>

A power law distribution has the form:

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101367106-c5736280-3873-11eb-90cd-29db53738d66.PNG" width="550px">
</p>

<br/>

When the frequency of an event (`y, x`) contrasts as a power of some marker of that variable's size[2], the frequency of that variable follows a power law. For example, the number of documents (`y,x`) in our corpus (`k`) have a certain size (`a`) and that size varies as a power to the size of the corpus. For our problem, each users words are grouped together in individual documents that make up a corpus derived from the `blues_traveler_2000` group. Therefore we can talk about the distribution of the number of words within each document of different corpora in the group. We can now begin to consider a distribution of the entire corpus which is made up of 302,731 documents distributed across 13 different users.

An important takeaway is that a small number of frequent words in each document are dramatically larger than a large amount of rare words. As we'll see with Zipf's law, a handful of the largest items will account for an obvious disproportionate percentage of the combined values of the overall distribution. 

<br/>

# zipf's law

In commonly used words, whether they be in a New York Times bestseller, an ancient script or a random newspaper article, a pattern emerges. The law that dictates this phenomena follows a discrete distribution that says that given a corpus of natural language, the frequency of any word is inversely relative to its rank[']. Thus the second most used word will appear half as often as the most used word. The third, one third as often, the fourth, one fourth as often and so on, which means the amount of times a word is used is proportional to 1/rank. The phenomena dictates that word frequency and ranking plotted logarithmically on the x and y axis of a graph will follow a straight line, which is governed by the power law, and its called Zipf's Law. 

Essentially this helps us understand how words are distributed across documents. Mathematically, Zipf's law states that if...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101367590-4fbbc680-3874-11eb-837d-ab01bf52b6fa.png" width="100px">
</p>

...is the most common term in the corpus and so on....

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101368167-d375b300-3874-11eb-99a3-cae01229101a.png" width="60px">
</p>

<br/>

...then the corpus frequency...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101368525-3f581b80-3875-11eb-8c1a-32b1ef8baf8f.png" width="100px">
</p>

<br/>

...of the...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101368724-7a5a4f00-3875-11eb-9339-3ded6943743a.png" width="100px">
</p>

<br/>

...most common term is proportional to...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101368966-c60cf880-3875-11eb-8413-6626f85e8021.png" width="200px">
</p>

<br/>

...so if the most frequent term in the corpus occurs...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101369163-fb194b00-3875-11eb-8db1-7e86199b59df.png" width="60px">
</p>

<br/>

...times, and the second most frequent term has half as many events, the third most frequent term a third as many events and so on. Basically the idea states that frequency decreases very rapidly with rank. Its surprising that something as complex as existence should be conveyed by something as imaginative as language in such a predictable way. Based on the idea of favorable attachment, i.e. *'the rich get richer'* process, if we generate word occurrences more likely of popular word types than of rare word types, the word frequency of the generated corpus will follow this law[3].

<br/>

# pareto distribution

According to Wikipedia, *"A probability distribution is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment"*. This isn't fake news[<a href="https://www.newsweek.com/riemann-hypothesis-million-dollar-math-mystery-michael-atiyah-1146244" title="Facepalm or headesk" rel="nofollow">'</a>], its true. Certain things that we do as humans are quantifiable and even mundane tasks are capable of having their own probability distribution. As an unrelated example, let's define the variable `X` as the number of times you pull a penny from a jar full of pennies with the probability of that penny being from the year 1995. Thinking at a lower level, what are the other possible values for this random variable? To figure this out we would need to plot every penny from the jar and see how the distribution is spread out among those possible outcomes.  

Bringing it back to the problem at hand, Zipf's law is known as the discrete version of the continuous Pareto distribution from which we get the Pareto Principle. The Pareto Principle tells us that its worth assuming 20% of causes (`bt_1` though `bt_4`, pictured in the orange bar plot above) are responsible for 80% of the outcome[4]. This is also known as the 80/20 rule and coincides with many discernible phenomena in social, scientific, and geographic settings.  

In language specifically, the most frequently used, 18% of words (stop words) account for over 80% of word occurrences. Which explains by way of the power law, why the relationship of common words are more frequent than rare words. It's also worth mentioning that there is a relation with the exponential distribution[5] as both are encompassed in the Exponential Family[<a href="https://en.wikipedia.org/wiki/Exponential_family" title="Exponential Families definition" rel="nofollow">'</a>].

The Pareto distribution embodies a useful power law. The spread of the distribution is most often presented in terms of its reliability function, which gives the probability of seeing larger values than `x`. The reliability  function of a Pareto distribution for...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101370076-091b9b80-3877-11eb-9310-11f57cf2fc70.png" width="200px">
</p>

<br/>

...is...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101370324-5c8de980-3877-11eb-9c0a-cc1106355a10.png" width="150px">
</p>

<br/>

The value of this reliability function is initially 1 and decreases to 0 as x increases. It defines a continuous probability distribution on...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101370569-a545a280-3877-11eb-91af-e2a17d5aa6a3.png" width="155px">
</p>

<br/>

...but we're only interested in:

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101370742-de7e1280-3877-11eb-9ca3-f3ed60bb1673.png" width="155px">
</p>

<br/>

Normally we would be interested in b > 1 which is required for the fixed mean value. We can then call...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101370928-2735cb80-3878-11eb-8826-a844770065a2.png" width="130px">
</p>

<br/>

...the site parameter; we call b > 0 the slope parameter; and the distribution is then:

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101371252-85fb4500-3878-11eb-820a-383388f83efc.png" width="200px">
</p>

<br/>

It's not obvious, but there's a similarity with Zipf's law. Both functions decrease exponentially, which explains the nature of the Pareto's fat tails. Fat tails on a distribution highlight outcomes that are far from the mean, and this is the result of the probability density function being proportional to a power function of the form: 

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101371475-cc50a400-3878-11eb-9ac8-d724cd1be94a.png" width="230px">
</p>

<br/>

where:

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101371670-03bf5080-3879-11eb-9cc8-8e85c0a44fe1.PNG" width="240px">
</p>

<br/>

The Pareto is typically used in finance, econometrics and physics to detail the distribution size of things like human settlements, the sizes of sand particles, standardized price returns on individual stocks, as well as to describe the allocation of wealth among populations.

In theory this distribution could be used to model our text. Researchers have explicitly used the Pareto distribution for word analysis before, and make no mistake we can probably say that the text is Paretian. The only necessary assumption we need to make is that if you're estimating the data from the tail and its from one of the several families of distributions that share the same tail, the tail of your observation only needs to deliver a value `x` on the vertical axis, `P(x)` on the horizontal axis (inverse for Zipf) and the tails sense of 'thickness' gives some indication to the probability of extreme events.

The Pareto distribution is continuous, but our main variable is a counted number of occurrences with each word as an occurrence, so the length of each word should be measured by a unit that respects language as a group of letters rather than just letters. Modeling the Pareto method would be useful if performing Bayesian Inference on the corpus was our goal, but again we're trying to prove a discrete pattern. Highlighting the Pareto simply strengthens our intuitions so that we're aware of it's connection to the next method, which is more relevant in identifying the distribution of this dataset. 

<br/>

# zipf distribution

The Zipf distribution, is the discrete version of the continuous Pareto distribution. The linguistic intricacies of text varies significantly with the size of the corpus, the domain/context, and the medium of communication, but it's safe to assume that the most frequent 150 words usually account for about half of the words in a corpus. This distribution provides a threshold model for the expected occurrence of target words, which can help us answer questions regarding a words role in the corpus.

It may not be immediately obvious why the mere occurrence or relative probability of a word is of significance to the Zipf, but we can use it to find the range of semantic influence of a word in a given corpus, reveal how the pattern of occurrences contribute to the assessment of its relevance to the corpus, and identify the most common stop words in a corpus, which are just a few useful things. That being said, the assumptions of the Zipf are but a small part of this analysis and is just a tool for an evaluative purpose. 

A formal summarization states:

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101372078-7cbea800-3879-11eb-90d5-7d8022ca151b.png" width="180px">
</p>

<br/>

Where `n`, a positive integer, is the number of words (elements), `k` is their rank and `s`--which is equal to or greater than zero--is the value characterizing the distribution, or the parameter that determines the shape of the distribution. The heavy-tails of this distribution can be modeled with an `s` close to `1`, and the formula is used to predict the frequency of the words of rank `k` out of a sum of `N` elements.

The collective number of occurrences of each user in the dataset appears synonymous with the head, curve and tail of the Zipf distribution. The word frequency and ranking plotted on the x and y axis of this logarithmically scaled graph follow a straight line, meaning it fulfills the assumptions of Zipf's Law[6].

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101372605-1423fb00-387a-11eb-8421-f6f2ef1d9502.png" width="500px">
</p>

<br/>

The word frequency in the plot above is upwards of 300,000 documents which make up the entire corpus. When you're talking about huge numbers you'll want to express things in scientific notation, and to do this we use logarithms.

For example, the frequency of this audio filter effect...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101372870-55b4a600-387a-11eb-832d-ddd1a7855946.gif" width="500px">
</p>

<br/>

...is measured in logarithmic units called Hz... 

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101373723-47b35500-387b-11eb-8eb0-596a2aedc490.PNG" width="300px">
</p>

<br/>

...which are represented by an exponential difference and its using this difference to scoop all of the low frequencies out of the sound so that only the highest frequencies remain.

This is interesting because if sound were measured linearly, the fader in the gif above would have a range of 1 to 1,135,910 notches next to it instead of 1 to 10, which would be ridiculous. By using a logarithmic unit, we can represent the range of 1 to 1,135,910 with only 10 notches next to the fader. So logarithms allow us to compare super large numbers with super small numbers.

The logarithm is also used to illustrate word frequency because exponential power laws occur naturally to frequency responses on Earth between many phenomena, which is obvious when plotted on a log-log graph. The `blues_traveler_2000` logarithmic plot from above says that each unit of distance from the lower left is 10 times the value of the previous unit. So the distance from 0 to 1 will appear the same as the distance between 1 and 10, which will appear the same as between 10 and 100 and then again between 100 and 1000. This type of scale will de-emphasize the total contrast in frequency among the most frequent words and sort out the nuanced differences among the infrequent words. 

We can also think of the logarithm of a number as how many bits we need to represent it. A single bit has two states, 0 and 1. Thus to represent two states we need a single bit, so:

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101379621-784abd00-3882-11eb-896d-7bfd50029d1b.png" width="200px">
</p>

<br/>

To represent four states, we can use 00, 05, 10 and 15. That's two bits, so:

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101379839-bfd14900-3882-11eb-8f2c-93bd78e96956.png" width="200px">
</p>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101380020-fb6c1300-3882-11eb-8bd3-677d383f0e2d.PNG" width="340px">
</p>  

<br/>

# Sampling

Processing large streams of text data for a dataset made up of 300k documents can be computationally expensive, so we're going to pick a random user, and distill their text into a state where it becomes easier to perform iterative preprocessing. Normally something like this would require the sample is a representative of the distribution that governs the data.

The possibility of using a sophisticated method of sampling is always present. We could use rejection sampling, systematic sampling or simple random sampling, but the odds of the semantic similarity of each users text being exactly the same is a naive assumption. I'm certain that each user's words, besides the commonality of lexical words are unique and do not need to be representative of the entire population. 

Given that I would like a fast, inexpensive and easy technique, I will randomly use `bt_4` as my initial sample as `bt_4` appears to be the perfect size. Not too big, not too small...

<br/>

```python
# sample: bt_4

bt_4 = dataset['Name'] == 'bt_4'

# shape of data
print('bt_4 data shape: ', bt_4.shape)
```

## `bt_4 data shape:  (38954, 4)`

<br/>

For this initial text preprocessing phase I'll use the spacy (NLP python library) pipeline feature to process 500 samples of `bt_4`'s text at a time to grab the sample word vectors and format them into numpy arrays. 

The first part of the code is used to clean the text by lemmatizing the words and removing personal pronouns,--in `spacy` the lemmatized string of personal pronouns is `'-PRON-'` --stop words and punctuations. We'll dive deeper into the granular intricacies of text preprocessing later on, so for the sake of brevity let's dive into the code. 

<br/>

```python
# create function to clean up text by removing personal pronouns, stopwords and punctuation

import spacy
nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation

def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        tokens = re.sub("(^|\W)\d+($|\W)", " ", tokens)
        tokens = re.sub('[^A-Za-z0-9]+', '', tokens)
        texts.append(tokens)
    return pd.Series(texts)
```

<br/>

The next few lines of code will obtain all the words from `bt_4`'s message feature and put them in a list. Then the text will be cleaned using the `clean_text` function, which will remove common stop words, punctuation and make all words lowercase. Next, all of the 's will be removed because `spacy` doesn't remove this contraction when lemmatizing words. Lastly the count occurrences of all words are gathered.

<br/>

```python
# collect all text associated to bt_4

bt_4_text = [text for text in dataset[dataset['Name'] == 'bt_4']['Message']]

# clean bt_4 text
bt_4_clean = cleanup_text(bt_4_text)
bt_4_clean = ' '.join(bt_4_clean).split()

# remove words with 's

bt_4_clean = [word for word in bt_4_clean if word != '\'s']
# count all unique words
bt_4_counts = Counter(bt_4_clean)
bt_4_common_words = [word[0] for word in bt_4_counts.most_common(30)]
bt_4_common_counts = [word[1] for word in bt_4_counts.most_common(30)]
```

<br/>

After the text has been preprocessed, the 30 most frequently occurring words for `bt_4` are visualized using `matplotlib` and `seaborn`.

<br/>

```python
# plot 30 most commonly occuring words

plt.figure(figsize=(20, 12))
sns.barplot(x=bt_4_common_words, y=bt_4_common_counts)
plt.title('Most Common Words used by bt_4')
plt.show()
```

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101381350-a4673d80-3884-11eb-8fc3-34363c72e294.png" width="600px">
    <img src = "https://user-images.githubusercontent.com/29679899/101381508-d2e51880-3884-11eb-8f52-cccf32cd373a.PNG" width="500px">
</p>

<br/>

Normally it's said that in the English language, the word *'the'* represented in any given medium (New York Times bestseller, an ancient script or a random newspaper article) will have about 6% the frequency of anything you will ever say, read or write, and we can say the same for `bt_4`'s use of the verb `lol`. Thanks to our stop words from `spacy` we were able to reduce the dimensions of words down from what would be considered globally common to something that's more locally common to this user and perhaps every user. The plot makes it pretty clear that certain dialectal language is abundant on social media platforms.  

So if the greatest frequency of words in `bt_4`'s sample are words that provide a shallow connection to ideas, but lack much of anything that can be revealed in terms of meaning for this sample, how can we successfully reduce the noise in this corpus down to something that can give us insight into what `bt_4` actually talks about, given that the core meaning of `bt_4`'s words are contained within information that would appear to be rare within the context of her corpus? I believe that we'll need to preprocess this sample even further and even create a custom list of stop words to reveal `bt_4`'s most infrequent words.

<br/>

# feature extraction

Whether we use shallow or deep methods, most of the learning tasks in this post can be boiled down to supervised learning. It requires us to define what the important concepts are for the problem and label those concepts to the data. In order to understand what feature extraction is, let's illustrate an example.

When looking at dates and times, they can be modeled using the exponential families, much like words and since the exponential distribution[7] is especially useful for describing events that occur randomly, we can use it to model the elapsed time between the occurrence of overlapping messages from a given user at any point in time. Let's think of these points in time as events.

A non-overlapping event could include the time between a user's response to the group, or the amount of responses at one time etc. This is to say that the distribution is able to describe the time--a continuous or discrete variable depending on our goal--between events that occur continuously and independently at a constant average rate. So theoretically, time could be an important factor in the response rate of each user. 

If extracting time into a feature is our goal, we can start to think of time spent between users conversing as an event. Then we can ask ourselves questions like... 

*At a high level, how long are events?*

*What is the general duration of events when there is a sparsity of users?*

*Are there patterns regarding the duration of events when certain users are present?*

Given this inference we can convert dates and times into a useable format by transforming them into numeric and categorical features. If we break them down we'll end up with something like: `Jul-28-2018:09:32`.

We can break the time-stamps down further by saying month: `July, day: 28, year: 2018, hour: 9, minutes: 30, seconds: 2`.

This provides 6 features that we can use to model our inference. If we wanted to treat these features as categorical, we could end up with even more features. For example, suppose we one-hot encode the observation of any given time in our hour variable. That would give us a total of 24 more features we could use. So based on these features we can use time to model any of the three questions from above.

<br/>

# feature selection:

I'll use a series of violin plots to visualize the distribution of each feature for every user, but before we dive in let's try to understand violin plots.

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101383779-8c44ed80-3887-11eb-9834-39dea1bd4bf1.png" width="300px">
</p>

<br/>

The shape of this particular plot represents a bimodal distribution and estimates the kernel density function of its given observation. The function's shape is created by something called nonparametric(kernel) density estimators, or KDEs, which smooth the observations probabilities. The outer shape of the plot graphically highlights the mode and symmetry of the data's distributional qualities[8].

Violin plots are useful to us because they they visually show us the most important numerical signals of each sample in the data. I like using violin plots over traditional, static box and whisker plots because they capture the data's distribution in a dynamic way. 

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101383346-01fc8980-3887-11eb-9a18-7fa13b23a202.gif" width="625px">
</p>

<br/>

Executing this code let's us visualize each users distribution.

<br/>

```python
# dependencies

import numpy as np
stopwords = stopwords.words('english') 

# number of words in the text 
dataset["num_words"] = dataset["Message"].apply(lambda x: len(str(x).split()))

# number of unique words in the text 
dataset["num_unique_words"] = dataset["Message"].apply(lambda x: len(set(str(x).split())))

# number of characters in the text 
dataset["num_chars"] = dataset["Message"].apply(lambda x: len(str(x)))

# number of stopwords in the text 
dataset["num_stopwords"] = dataset["Message"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))

# number of punctuations in the text 
dataset["num_punctuations"] =dataset['Message'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# number of upper case words in the text 
dataset["num_words_upper"] = dataset["Message"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

# number of title case words in the text 
dataset["num_words_title"] = dataset["Message"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

# average length of the words in the text 
dataset["mean_word_len"] = dataset["Message"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
```

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101384713-c9f64600-3888-11eb-91e6-52107251dd9b.png" width="500px">
</p>

<br/>

With the values of each feature defined on the y axis, we can see that this feature is not distributed normally. Notice how varied the number of words are per user. Given the varying number of word counts per users corpora, this feature could provide a learning algorithm with useful information to differentiate one user from another.

We can also see how different `bt_13` is from the rest of the users. This is very important because we can compare the frequency of features for each user against other users, in turn allowing us to determine the significance of each feature by user and vice versa.

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101384959-2194b180-3889-11eb-8c8a-c09cf4dcc0d8.png" width="500px">
</p>

<br/>

The number of unique words in each users corpus appears to be reasonably distributed, minus `bt_11`, `bt_12`, and `bt_13`.

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101385192-6fa9b500-3889-11eb-8b90-d88853d1022b.png" width="500px">
</p>

<br/>

Another basic feature we could extract is the number of characters used per users message. This feature is pretty even, minus 4 users. Notice the interquartile range of `bt_9` and how it's slightly higher than every other user...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101386029-93212f80-388a-11eb-8afb-3a5d098eae3f.png" width="500px">
</p>

<br/>

...and showing a really fat head and tail on its lower quartile. The interquartile range is used to find the difference between the middle of the first half of the observations and the middle of the second half of observations, so its a measure of spread or how far apart the data points are.

If we're taking social media conventions into account based on our second observation of Zipf's law, we can infer from `bt_9`'s character count that he produces more characters in his documents than any other user. Highlighting that `bt_9` is a more conventional communicator i.e. `bt_9`'s text is less prone to typical social media conventions. Looking over each users IQR...there's a slight variation of the mean in every single observation denoting another possible feature. 

Notice how `bt_13`'s interquartile range is over stretched... 

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101385935-738a0700-388a-11eb-9c95-b45d8e683de1.png" width="500px">
</p>

<br/>

...even representing negative values. The estimation of the KDE is influenced by the sample size. Furthermore, violins with a relatively small sample size almost always appear misleadingly smooth, thus indicating `bt_13` as having the lowest occurring value in the corpus and can be treated as an outlier.

Albeit the variance is not as great as the number of words in each users corpus, the number of stop words by user might be a good feature.

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101386222-df6c6f80-388a-11eb-94c5-43050de3db69.png" width="500px">
</p>

<br/>

5 out of 8 users have varying degrees in their usage of punctuation. This may be a relevant feature, but I'm not convinced. 

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101386351-117dd180-388b-11eb-9746-cba28abb1aaa.png" width="500px">
</p>

<br/>

The variance of upper case words by each user is promising. 

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101386610-67527980-388b-11eb-86a9-c744e871ffa9.png" width="500px">
</p>

<br/>

`bt_2` looks drastically different from the rest of the users, so let's take a closer look...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101386744-9b2d9f00-388b-11eb-9593-964237e5fdfc.png" width="500px">
</p>

The long tail on the upper quartile is a measure of the data’s spread or the number of values that the feature possesses, and the spread of `bt_2`'s text covers a wide range. When you have a shape as squished as `bt_2`, it represents a non-symmetric distribution and we must asses the characteristics of its shape in comparison to other possible features to determine how useful the feature.

To consider the skewness of `bt_2`, closely observe the value of the lower quartile in comparison to the upper quartile. A good portion of observations are compressed to the 0 - 0.45 range. We can infer that anything less than 4 1/2 *IQR below Q3 appears to be an outlier*. So the majority of `bt_2`'s upper case words lie in the 3rd and 4th quartile range with the outliers of lower case words residing in the 1st and 2nd quartile ranges. 

The lower quartile has between 1 and 6 varying sizes of peaks, indicating a non-symmetrical multimodal distribution, so `bt_2` has multiple modes (distinct peaks or local maxima) in the probability density function. `bt_2` could be multimodal for a number of reasons. There could be patterns or the user may have a preference toward using upper case words, which in the context of social media is the equivalent of shouting.

This could also be a sign of an overlapping distribution, which means a more distinct pattern could lie within this feature, leading me to believe that the signal needs to be further deconstructed. For example, words that are not actually uppercase could somehow be leaking into words that are uppercase creating a noisy feature. 

With all of this in mind its easy to assume that most if not all of the data is multimodal, so it's hard to say whether or not any features extracted from the users will be good, "off the shelf" predictors. We can explore the user features in a higher dimensional space to explore this assumption.

Another relevant feature produced by the number of title case words.

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101387175-155e2380-388c-11eb-88d7-3d8181a5dc94.png" width="500px">
</p>

<br/>

Lastly, the average length of words produces nothing relevant.

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101387305-43436800-388c-11eb-9247-f2aab2d53519.png" width="500px">
</p>

<br/>

There's absolutely no way we can talk about feature extraction without acknowledging sentiment. The main reason sentiment analysis is so difficult is that words often take different meaning depending on the domain in which they are being used. This is why out-of-the-box analysis tools to categorize sentiment across many domains do not exist.

Be that as it may, my basic intuitions regarding this dataset dictate that funny, negative and sarcastic sentiment contains slightly more words than positive sentiment. But in the same breath, various forms of a single word will be associated with varying sentiment and not all words are used in their literal context. 

To illustrate this example, let's turn our attention to the parts of speech... 

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101387676-c9f84500-388c-11eb-8f23-764b9e67c035.png" width="400px">
</p>

<br/>

...used in a few text examples derived from varying points in the corpus. Notice how the tagged parts of speech are similar in these two sentences...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101387946-396e3480-388d-11eb-96c9-6bd58e72ef42.png" width="400px">
    <img src = "https://user-images.githubusercontent.com/29679899/101388090-7b977600-388d-11eb-8132-9a73c2e89a61.png" width="600px">
</p>

```python
1st sentence character count: 19 
2nd sentence character count: 38
total: 57 sentences primarily made up of adjectives and verbs
```

<br/>

...and these two sentences also share their own similar sentiment pattern...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101388350-d16c1e00-388d-11eb-9488-b1b9bc7c65ae.png" width="680px">
    <img src = "https://user-images.githubusercontent.com/29679899/101388381-ddf07680-388d-11eb-8de2-bbfb7d2b54b8.png" width="400px">
</p>

```python
1st sentence character count: 92 
2nd sentence character count: 26
total: 118 sentences primarily made up of prepositions and nouns
```

<br/>

...a rather naive pattern at this point because all four of them share similar parts of speech, but based on this simple intuition, we may be able to reinforce these part of speech priors and weigh them against sentiment in some meaningful way during topic modeling.

The parts of speech used in the corpus are obviously varying for each user, denoting 6 more possible features that we can use, bringing our final total number of handcrafted features for each user to 11:

<br/>

```python
number of words in text
characters and special characters in the text
stop words
upper case words
title case words
nouns 
prepositional phrases 
adjectives 
verbs 
personal & possessive pronouns 
existential words 
```

<br/>

# 3. cleaning & refining the sample

 
In this section, `bt_4` will undergo further preprocessing to reduce her corpus from its previous shape of `(38945, 4)`--which still contains a considerable amount of stop words--into something easy to visualize so that we can begin passing the text through the proposed shallow and deep learning models.

The dependencies for this section will include a few that we're familiar with like, `numpy` and `pandas`. Then we have `re`, `PorterStemmer` (removes morphological affixes) and the stopwords module from `nltk`, python's custom `stopwords` list, and another `STOPWORDS` list from the `gensim` module. The combined strength of our new stop word lists contains 3x the common filler words used in the list from the previous sampling section.  

```python
# dependencies

extra_stopwords = []
bt_4_additional_stopwords = extra_stopwords

import nltk
import gensim
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer 
from gensim.parsing.preprocessing import STOPWORDS
```

Another stop word list was included inside of the extra_stopwords variable (which will remain unseen), which contains a custom list that takes a lot of dialectal social media language into account that is specific to each user. So words like `bcuz, www, cuz, kno, nah, tht, woof, tho, irl` etc. will be excluded.  

To initiate the second phase of text preprocessing, we need a `for` loop to iterate over all of the remaining words in `bt_4`'s corpus. Before we write the loop, the corpus variable needs to be established so that at the end of the loop, all of the cleaned text can be appended to the new corpus list. To initialize the `for` loop iterations, the range of the loop will be equivalent to the number of observations within `bt_4`'s corpus `(0, 38954)`. 

<br/>

```python
corpus = []
for i in range(0, 38954):
    clean_text = re.sub('[^a-zA-Z]', ' ', str(bt_4_text[i]))
    corpus.append(clean_text) 
```

<br/>

When looking at the previous code examples, while you may be aware of what regex (regular expression) commands may look like `(re)`, lets dive deeper into what they are and how they're helping us preprocess `bt_4`'s text. 

<br/>

# regular expressions
 
Or regex, is basically a specialized text based programming language native to python that allows you to define text patterns. With this pattern you can use regex commands to find or find and replace text. Simply put, you can define instructions for the set of possible strings that you want to match, and you can change them by deconstructing them in various ways. 
 
In this alphanumeric example, the first visual field is a representation of the regular expression and will show us how a pattern is processed by the regex language. The first text field represents the regex that we will write and the second text field represents the text that we would like to match, and will be manipulated by the regex in the first text field.

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101543670-88da6080-3972-11eb-85dd-814c1b1cbab8.png" width="650px">
</p>

<br/>

The caret symbol in the first line of the first text field will match the beginning of the expression `[^a-z]` to whatever it is that we want to specify, which are the letters A through Z. We're explicitly telling regex that if all of the characters in the sentence that we want to match in the third box fall under a pattern of being the first lowercase word in a sentence and contains the letters A through Z, that a successful match on the target text has been made. 

After regex matches our argument to the target text, I'll place a pipe symbol `'|'` after the first expression and define a second argument `[_A-Z]` that will match Unicode word characters; this includes most characters that can be part of a word in any language, as well as numbers and the underscore symbol. Another pipe symbol is used and the last expression `[0-9]` matches any Unicode decimal digit, which there is clearly a digit in our sentence. The regex matches the targeted digit concluding our search pattern definitions. 
<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/101543887-e078cc00-3972-11eb-9db0-e842c118de4d.png" width="450px">
</p>
<br/> 
The pattern of the regular expression as seen in the example are used to specify a set of strings that match our text preprocessing requirements and all three patterns match all of the characters within the target string, which tells us that all three or a variation of each expression can be used when cleaning `bt_4`'s text of unicode characters, numbers, emoji, punctuation and capitalization. 


<br/>

# cleaning the sample

To illustrate how each cleaning technique works, at random we'll sample the 1000th document from `bt_4`'s corpus. 

```python
clean_text = bt_4_text[1000]
```
## `Goddamn season 2 of Queer Eye`
<br/>

The `re.sub` method contained within the first `clean_text` variable of our upcoming for loop will eliminate all characters in `bt_4`'s messages except for letters.

This means we'll be removing numbers, punctuation, emoji and keeping all letters from `A - Z`. It's debatable whether removing the punctuation in our text will increase or decrease positive affects from learning, but as this analysis is broadly exploratory, I do not think keeping the punctuation is necessary. 

The first parameter of the sub function will specify what we don't want removed from the text, so all characters from `A - Z`. This is done by placing all characters from A to Z in upper and lowercase format inside of the regular expression: `'[^a-zA-Z]'`. `bt_4`'s messages will be another parameter for the function to specify to the regex where we want the characters from A to Z to remain.
 
When we remove characters from the 'Message' feature vector, the two characters that are at the left and right of the character being removed will end up sticking together and possibly form a nonsensical compound word. So we'll input `' '` as another parameter for the sub function to replace the character that's being removed, by another character which will be represented by a space. 

```python
clean_text = re.sub('[^a-zA-Z]', ' ', str(bt_4_text[i]))
```
## `Goddamn season  of Queer Eye`

<br/>

For the next step of the cleaning process we will lowercase all of the letters in each document of the corpus. We'll simply take the `clean_text` variable and on the other side of the equal operator call the same variable but add the `.lower()` method to the variable which takes a copy of a string as input and returns all lowercase strings.

```python
clean_text = clean_text.lower()
```
## `goddamn season  of queer eye`

<br/>

Next, we need to transform the output `'goddamn season  of queer eye'` so that its not represented as a sentence. Why would we want it to be anything other than a sentence? Each individual sentence within the corpus is composed of something called a token. Tokens are individualized representations of each word in the sentence. So each word in the sentence will stand on its own, but remain in the same sequence.
 
Tokens are not word agnostic, they can include punctuation, emoji's, and ASCII characters. If some individual component of the document is important and can be a contributor to parsing the text, then we will use that component as a token, which leads us to the process of tokenization. When you have a document that you want to build as a bag-of-words model, its necessary to split each sentence up into a list of words which makes it easier for the model to process. In python this is done with the `.split()` method which we'll attach to the `clean_text` variable. 

```python
clean_text = clean_text.split()
```
## `['goddamn', 'season', 'of', 'queer', 'eye']`

<br/>

The `clean_text` variable is now a list of 5 elements, each element being an individual word or token that makes up this document. We can now create a series of for loops for our imported and custom stop words to go through the different words of `bt_4`'s list of tokens and remove the irrelevant words. This will help us retain only words that contribute to a sense of meaning within the sentence.

Reducing the tokens morphological variation gives the models process of learning a certain edge. Simply, morphology is the structure of a word whether it has an upper case at the beginning of the word, what affixes it has--ing, ed--past, present and future tenses etc. A given words morphology can be reduced by lower casing all of the words, which is necessary because we do not want the capitalization of a given word to matter, so that easier matching between words is possible. They can also be reduced by removing affixes from words and transforming the word into its base word or stem.
 
Stemming involves transforming words like remembered, remembering, remembrance into just the word remember so that if a document has a word that is important but not in the morphological construct that its presented in, it still counts. A similar concept to stemming is lemmatization. Previously mentioned in the sampling section, lemmas are used to understand the semantics of a word to find the pure root of a word rather than just removing a words tense. Lemmatization is a more computationally expensive method than stemming and due to my limited compute power, I will stick with stemming for now. We will add stemming to our main `for` loop using the `PorterStemmer()` method called from the `ps` variable and this will add stemming to all of the (words) in the message feature vector. 

```python
ps = PorterStemmer()
```

When constructing the stop word for loops, we'll need to go through each word in every document and look for all stop words which we've indicated in the `stopwords`, `get_stop_words`, `STOPWORDS` and `bt_4_additional_stopwords` variables. Each individual token in the `clean_text` variable will be included in a list `[word for word in clean_text]` function and the stop word conditions inside each stop words list will remove all of the words in the nested `clean_text` variable after the `if not` statement is called.

```python
clean_text = [ps.stem(word) for word in clean_text if not word in set(stopwords.words('english'))]
clean_text = [word for word in clean_text if not word in set(get_stop_words('english'))]
clean_text = [word for word in clean_text if word not in STOPWORDS]
clean_text = [word for word in clean_text if word not in bt_4_additional_stopwords]
```

<br/>

With our example illustrated, now we can put everything into a nice `for` loop that will iterate over every document in the corpus. 

<br/>

```python
corpus = []
for i in range(0, 38954):
    clean_text = re.sub('[^a-zA-Z]', ' ', str(bt_4_text[i]))
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    # text stemming & stop word removal
    ps = PorterStemmer() 
    clean_text = [ps.stem(word) for word in clean_text if not word in set(stopwords.words('english'))]
    clean_text = [word for word in clean_text if not word in set(get_stop_words('english'))]
    clean_text = [word for word in clean_text if word not in STOPWORDS]
    clean_text = [word for word in clean_text if word not in bt_4_additional_stopwords]
    corpus.append(clean_text) 
```

<br/>

The `corpus` is initialized as an empty list `[]`, the `i` after `for` will be the index going through all of the documents in the `'Message'` feature vector. Next the `range` will be specified for all of the values that `i` is going to take, so from from `0` to `38,954` will be indexed by `i` and is going to `.lower()`, `.split()`, `PorterStemmer()` the text and remove the stop words for each document in the dataset. Lastly the new corpus will be appended to the original empty corpus list. 

<br/>

# word2vec

For as smart as computers seem, the easiest way to understand what they do is to become as stupid as they are. This is a bit tongue in cheek, but when comparing how a 2 year old can comprehend and understand language and the tremendous effort that scientists and engineers must go through so that a machine can do the same is pretty astonishing. When a 2 year old is learning to speak and recognize letters, she's learning what are called symbolic expressions.
 
Symbolic expressions are what we see when we look at words. The idea that our thoughts are made up of a symbolic language, just because that's what we see, say and imagine is naive[9] because as you read my words right now, photons analogous with the patterns of each letter travel through your retina, setting off electrical cues that travel along the thread of an axon, which releases chemicals into the synapses of neurons, which travels to other neurons. This sounds like a series of abstractions following a complex sequence, and maybe such a thing can be described or recreated with the help of mathematics. So in order for computers to understand language, maybe we can transform our words into a numerical representation. 
 
Learning continuous characterizations of words has a deep history in natural language processing (*Rumelhart et al., 1988*), but we need a way to represent words in such a way that they carry semantic meaning. 

For example if the cat name `cobra` is word number 5,391 in a corpus, you can represent the name cobra as the number 1 inside of a vector in the 5,391st position of the vector. With 0 outside of the vector being the notation that denotes the vector as an nth dimensional embedding. So we're representing words inside of a vector with a few thousand dimensions.

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/102142727-e36f3300-3e30-11eb-8df7-f1738b305ab1.png" width = "70px">
</p>

<br/>

`cobra` is just a single vector so let's create a series of vectors. We'll represent the cat name `baby cat` as the number 1 in a vector as the 9,853rd word...

<br/>

<p align="center">
    <img src = "https://user-images.githubusercontent.com/29679899/102142867-16192b80-3e31-11eb-8cea-c94c1e7ea74d.png" width = "80px">
</p>

...and so on...

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102160615-c6982700-3e53-11eb-954e-d5b8e7bf0327.png" width="700px">
</p>

<br/>

One of the drawbacks of representing words this way is that it treats each word as an entity of itself and it doesn't allow the model to easily discern words. 

If the model's seen the sentence *"cobra is licking her fur"*, even if its learned that the most probable outcome of *cobra licking* is *her fur*, if you give the model another sentence *"baby cat is licking her fur"*, as far as its concerned the relationship between `cobra` and `baby cat` is no closer semantically regarding the relationship between `dolly`, `basil` and `colin powell chain`. 

Its not easy for the model to generalize that *licking* and *fur* are common words when used in relation to *cats* and that the possibility of *baby cat licking her fur* is also just as likely as *cobra licking her fur*.The reason this happens is because the inner product between any two vectors is 0 which means the Euclidean distance between any pair of vectors is the same. 

Instead of doing things this way, we can represent all of our words as an embedding matrix. 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102289172-06224a00-3f0c-11eb-9b49-5479109ad20e.png" width="800px">
</p>

<br/>

For every cats name we can learn a set of features and values per feature. For example we can denote the cleanliness for each cat between the values `-1` for very dirty and 1 and very clean, with other values deviating from either number to provide contrast to each cats cleanliness. Age can be represented in a way that says each cat is neither young nor old and simply give them a value that lets us look at each cats age as a structured continuous variable. The cats gender can be simple binary values, `0` for male `1` for female.

Lets say for the sake of illustration our total number of features ends up being `100` which includes information like the cats domestic breed and whether or not they have short or long fur etc. `C` at the bottom of each vector allows us to take the list of numbers from each column that now represents the cat names as 300-dimensional vectors.

If we use this representation to represent the cats cobra and baby cat in the sentence, notice how we can now say that each cat is similar or dissimilar in some way. Their values will be different but a lot of features for each cat have some level of similarity. 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102289481-c9a31e00-3f0c-11eb-9040-ea7e4836dd0f.gif" width="675px">
</p>

<br/>

This increases the odds of the model generalizing that its just as likely or unlikely for colin powell chain to be as dirty as baby cat or cobra. 

This simple example does a lot to illustrate the intuition behind word representations but we can't expect the text in our dataset to be so structured. To get ahead of this we can learn target/context pairs. Let's take the sentence from our previous example:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102289656-31596900-3f0d-11eb-9d14-94c2ab440f87.png" width="265px">
</p>

<br/>
 
The name `cobra` is the target word and `licking her fur` is the context.

To learn an embedding for a sentence, it needs to be arranged in a way that derives context from the words on the left and right of the center word in the sentence. To solve this, we will train a word2vec model and we will only make use of its hidden layer weights, which are represented as an embedding matrix of the words based on their context in the sentence and then use gradient descent by way of backpropagation to correct its loss.

There are 2 types of context prediction methods. You can pick a word at random and try to predict the left and right adjacent words. So we have the sentence, *"That phat cat dolly sat on the mat"*, in order to predict all the words that surround `dolly` using the word2vec architecture, we can define a continuous bag of words or a skip gram model.

If we're at the word `dolly` in our window, we'll need to look at `'The phat cat _______ sat on the mat'`, as our input and try to predict the missing word `dolly` as the output. This is called a continuous bag of words (CBOW), and its non-linear hidden layer is eliminated and the computed weights are divided among every word in the sentence. 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102290040-2a7f2600-3f0e-11eb-8a15-8e8eb5c371e6.png" width="450px">
</p>

<br/>

To predict the current word based on the context, every word in the sentence gets estimated into identical spots and their vectors are averaged. Also the past input data does not affect the projection of the weights and it can even use words from a future sequence. Its training complexity is:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102290482-21428900-3f0f-11eb-973c-1ee7abd6667c.png" width="450px">
</p>

<br/>

We can also use each word as an input to a log-bilinear classifier with a continuous projection layer and predict words within a certain range before and after the current word[10], resulting in a continuous skip gram model:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102291029-32d86080-3f10-11eb-8d70-a2ff26ae08da.png" width="450px">
</p>

<br/>

The skip-gram predicts the surrounding words given the current word. The training complexity of this architecture is proportional to... 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102290818-cb221580-3f0f-11eb-8894-5e1d6ba84440.png" width="450px">
</p>

<br/>

...where `C` is the maximum distance of the words. If we choose `C = 7`, for each training word we will select randomly a number `R` in range `< 1;C >` and then use `R` words from the historical context and `R` words from the future of the current word as correct labels. This will require us to do `R x 2` word classifications, with the current word as input and each of the `R + R` words as output. In this experiment the maximum distance of words or window that we'll use is `C = 7`[11].
 
Both methods undergo backpropagating the loss function over and over again on the architecture until a table of word embeddings are accumulated. A loss function being a specific group of instructions that basically asks the network, *"Hold on a minute, what's the cost of this gradient computed by the weights of our forward pass? If the cost value does not meet our desired threshold, we need to send the weights back through the skip-gram algorithm to learn better"*. Without the loss function there's nothing to inform the network how wrong it is. When this process is complete we end up with an embedding matrix. Every word is a row, every column is some nebulous embedding dimension and each word in space holds significance.

The skip-gram method will end up looking something like this: 
 
<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102291589-6ff12280-3f11-11eb-8af5-79954ab4193c.png" width="700px">
</p>

<br/>
 
<code>w<sub>t</sub></code> representing the center word `dolly` inside of a sparse vector, and `W` is a matrix representation of the center words. If we multiply the vector by the matrix...
 
 <br/>
 
 <p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102291741-bfcfe980-3f11-11eb-9344-99e7e11a7e86.png" width="200px">
</p>
 
 <br/>
 
 ...we'll get a representation of the center word. Our matrices denoted by...
 
 <br/>
 
  <p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102291841-f4dc3c00-3f11-11eb-91be-6b86c2f3d838.png" width="80px">
</p>
 
 <br/>
 
...are the same matrix for each position and store the representations of the context words. For each position in the context denoted by the six vectors on the right, we will multiply <code>W<sub>w<sub>t</sub></sub></code> by the matrices and end up with the dot product of the center word with each context word. We will then use the softmax function on the dot products to generate a probability distribution which will enable us to predict the probability of each word appearing in the context given that the target word is the center word.

But it can also return a ground truth for the context word denoted by the prediction vectors. So if the ground truth prediction in <code>W<sub>t-1</sub></code> is represented by a `1` in the right most vector that we'll say is the word `phat`, and a probability estimate of `0.1` is given to that word which could be the basis for a poor prediction which would predicate some loss in the model because our input center word is the word `dolly`.
 
As an aside, the softmax function...

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102418764-624ca300-3fcc-11eb-8efa-9f1ee13a5c65.png" width="500px">
  <img src = "https://user-images.githubusercontent.com/29679899/102418950-c4a5a380-3fcc-11eb-823c-176c502e6898.gif" width="600px">
</p>

<br/>

...simply interprets the dot product vectors as a probabilistic representation by squishing arbitrary real values derived from the matrix `[M]` into natural values. Arbitrary denoting various values assigned from `[M]` but remain unaffected by the changes in values from the softmax function. 

Whichever method is used, the word embeddings will be able to predict a given word based on its neighbors within a sentence. To reiterate, the algorithm has internal embedding vectors for words, and when the algorithm is training its trying to predict which words are occurring based on the internal embeddings of the algorithm and context of a given sentence. It's basically taking the hidden layer weight matrix and building a lookup table of words as they relate to each other.

Earlier we defined an embedding matrix example comprised of cats, but let's generalize the idea a bit. `bt_4`'s corpus has roughly 30,000 words and we would like to learn an embedding matrix `N` which will be a 100-dimensional by 30,000-dimensional matrix.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102419343-8e1c5880-3fcd-11eb-92b7-8ec324245d22.gif" width="500px">
</p>

<br/>

The columns for the matrix will be the different embeddings for the 30,000 different words in the corpus. Let's say the word `licking` is the 5,527th vector in the 30,000 word corpus. 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102419652-392d1200-3fce-11eb-9cf9-cc1a1ae7ad73.png" width="180px">
</p>

<br/>

The notation <code>0<sub>5527</sub></code> indicates a sparse vector with the number 1 in position 5,527 which is also a 30,000th-dimensional vector as tall as our original matrix above is wide. 

If the embedding matrix is signified by `N`, you can take `N` and multiply it by the 30,000th-dimensional vector... 
 
<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102419835-a3de4d80-3fce-11eb-8492-6bc18dfb7326.png" width="200px">
</p>

<br/> 
 
 ...and it will become a 100-dimensional vector. So N is `(100, 30k)` and O is `(30k, 1)` so the inner product will be a 300-dimensional vector by `1`. To compute the first point of this vector, you would multiply the first row of the matrix `N` with the vector <code>0<sub>5527</sub></code>. You end up with a lot of 0's multiplied by each other because of the sparsity of the vector, but when you finally get to the 1 in the 5,527th place, you end up with the first point of our `(300, 1)` vector which would be the word `licking`. 

Our next point mat is multiplied by the `(300, 1)` vector following the same order of operations as `licking` and the process is repeated for every word. Then we get our first embedding. We can think of this process as:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102419973-f881c880-3fce-11eb-9add-b371856977be.png" width="300px">
</p>

<br/> 
 
So we're training the predictive model on the corpus, but more importantly we're making use of the internal structure, or the embedding vectors inside the model which represent the words in the sentence. By comparing these embedding vectors using linear algebra, the algorithm will be able to derive context between words that show up in specific contexts over and over again.

The goal of using word2vec will be to learn the embedding matrix `N` by initializing `N` randomly and using gradient descent to learn all of the parameters of the given matrix. Training the word2vec model is simple. We'll call the `Word2Vec` module from the `gensim.models` class. The `bp2Vec` variable contains our module and several parameters that will define our model. The sentences iterable represents a list of lists of tokens, and size defines the dimensionality of the word vectors.

```python
from gensim.models import Word2Vec

bt4Vec = Word2Vec(sentences = corpus, size = 100, window = 5, min_count = 4, 
                  workers = 8, sg = 0, iter = 30, alpha = 0.020)
bt4Vec = bt4Vec.wv
```

`size` reduces the large dimensional vectors down to smaller vectors, which is the same as saying the number of dimensions of the word vectors will have some `n` number of columns, and `n` is going to be the amount of generalization that you want to reduce the word vectors down to. Word vectors/embeddings with smaller dimensions means more general but less accurate word representations and high dimensional word vectors/embeddings means less general but more accurate and potentially overfit. This parameter will require a bit of tweaking.

To better understand what's happening, imagine a set of words in Euclidean space represented as points--dots all over the place in 3D--that are stored in vectors in a high dimensional space. 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102697341-133c8300-4203-11eb-85dc-909936d64271.gif" width="500px">
</p>

<br/>

`window` will represent the max distance between the current and predicted word within a sentence as they occupy this space. This is important because looking at words to the left and right of a word enables us to identify documents based on their auxiliary semantic features. 

We'll also throw away 80% of the dots by getting rid of stop words using the `min_count` parameter which ignores all words with a total frequency lower than `4`. Now we can find the best match of all those vectors that remain after getting rid of the words or documents that do not match or carry latent meaning.
 
`workers` simply optimizes the speed in which the algorithm trains. `sg` defines the training algorithm as a CBOW or skip gram model, which is another parameter we'll tweak. `iter` controls the number of epochs over the corpus, where 1 epoch is equal to one full training cycle on the training data and we will set the algorithm at `iter = 30`. The learning rate `alpha` controls the convergence of the loss function to its minimum and will be set to `0.020`. 

The loss function being optimized in word2vec is negative sampling or NEG. NEG is defined by an objective that is used to replace every...

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102697616-3405d800-4205-11eb-9b1e-85e447b0913f.png" width="80px">
</p>

<br/>
 
...term in the skip gram objective[12]. So the task is to distinguish the target word...

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102697660-821adb80-4205-11eb-84df-2ff22bd073e2.png" width="90px">
</p>

<br/>

 ... which draws from the noise distribution...
 
 <br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102697673-98c13280-4205-11eb-9e8c-ee5470023d95.png" width="140px">
</p>

<br/>
 
...using logistic regression... 

 <br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102697686-b2627a00-4205-11eb-9327-8bb6c2306091.png" width="700px">
</p>

<br/>

 
...where there are `k` negative samples for each data sample.
 
<code>Q<sub>θ</sub>(D=1|w<sub>t</sub>,h)</code> is a binary logistic regression probability under the model of seeing the word `w` in the context `h` in the dataset `D`, calculated in terms of the learned embedding vectors `θ`.

Using logistic regression, the objective is able to tell the difference between a small percentage of the high probabilities to real words and low probabilities to noise words that word2vec assigns. We use such a small percentage because of the sheer amount of our training samples and we need lots of training samples because of the large weight matrices computed by the word2vec model. Large training samples and large weight matrices results in lots of computation which slows down the training speed. We obtain fast training from `NEG` because computing the loss scales only when the number of noise words that you determine `k` are not all of the words in the lexicon of real words `V`, so negative sampling basically boils the loss function down to a classification task, sifting through real words and noise words until it can learn the differences between them.
 
There are a handful of similarity metrics that can be used for word2vec vectors to determine how similar two data points are, but we will use the cosine similarity. This can be thought of as the angle between vectors:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102697598-f4d78700-4204-11eb-837c-40fb733f52df.gif" width="550px">
</p>

<br/>

I like to think of the cosine similarity as *"Do these observations have the same vibe?"* or *"are these things semantically similar?"*. Even if two documents have the same vibe or semantic meaning, their distance in physical space would be drastically different if their semantic difference is similar. So the cosine similarity is the same because the angle between your query and the documents are the same for each case, but their physical distance is very large, which is why we wouldn't use Euclidean distance here. 

After training the skip-gram model, we can look up words that are similar to other words in the corpus with the `most_similar` function, which returns the top 10 most similar words. 

<br/>

```python
bt4Vec.wv.most_similar('cool')
```
<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102835710-7c6bf400-43c5-11eb-9d39-4ff2ae41a72a.PNG" width="300px">
</p>

<br>

The scale of 1% to 100% next to each word represent the cosine similarity being least or most similar to the original query in the `most_similar` function.
 
The word `cool` can represent a state of being, aesthetic appeal, a behavioral characteristic, or can be used in non-commital phrases. So it makes sense that `cool` is highly similar to words like `nice`, `wow`, `ok`, `chill`, `sweet` and `good` which are words commonly used in the same context.  
 
One of the useful things about similarity is that you can follow hierarchies of words until you reach a theme or possible topic. For example...

<br/>

```python
bt4Vec.wv.most_similar('failure')
```
<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102839844-77ac3d80-43cf-11eb-9001-07ef1c0623ec.PNG" width="300px">
</p>

```python
bt4Vec.wv.most_similar('anxiety')
```
<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102839972-be019c80-43cf-11eb-88b8-ac89cf24823a.PNG" width="300px">
</p>

```python
bt4Vec.wv.most_similar('curse')
```
<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102840070-f012fe80-43cf-11eb-8673-4dae78726bbf.PNG" width="300px">
</p>

```python
bt4Vec.wv.most_similar('smartest')
```
<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102840173-3bc5a800-43d0-11eb-967f-b27ad1bd00f9.PNG" width="300px">
</p>

```python
bt4Vec.wv.most_similar('insane')
```
<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102840210-57c94980-43d0-11eb-85d3-8fdec63bb33f.PNG" width="300px">
</p>

```python
bt4Vec.wv.most_similar('loser')
```
<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102840249-6adc1980-43d0-11eb-81a2-a385af89e84b.PNG" width="300px">
</p>

<br/>

...with the cosine distance of each randomly chosen word staying close to 100%. We can assume the semantic similarities between each word are significant, but there's no way we can say that each word is causal. We just can't make that kind of assumption, but if we could divide `bt_4`'s text into topics, maybe we could gain a broader understanding of `bt_4`. 

In order to represent the text using Euclidean distance in higher dimensions, let's consider the previous shape of `bt_4`'s data. A `38954,4` array. After removing common words, the array's size was reduced to:

```python
print('bt_4 corpus length:', len(corpus))
```
# `bt_4 corpus length: 31413`

Each token in the corpus can be thought of as an array of text, with each discrete symbol corresponding to an embedding vector inside the model. The embeddings are contained by one-dimensional vectors, with each vector corresponding to a word in the vocabulary. Each word is mapped to vectors of real numbers, so that's 31,413 tokens and each one will contain several of their own embeddings and is expressed as the sequence in which the word occurs, but as a vector. 
 
To visualize the one-dimensional embeddings, we'll need to transform them into two-dimensional tensors. A tensor is a variable that has `n` indices where each index covers a range of dimensions of the three-dimensional space. In short, they are abstractions of scalars that give us a good framework to represent our one-dimensional vectors in Euclidean space.

<br/>

# restatement

I started this version of the project, so all of the work in this post, in August of 2017. It was completed in April of 2018 and was posted to my old blog a few months after that. The platform I was using at the time to host my content had many problems and in November of 2020 I lost the next 5 sections of the project. As mentioned in the introduction, the next sections would have included working with deep learning based algorithms to classify each member of the dataset to their respective text where the label was lost. 

Section 4 detailed the mathematical and lay intuitions on how to use the t-SNE algorithm to visualize word embeddings derived from one-dimensional vectors from word2vec and transform them into tensors to build semantic representations of the dataset....

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103373073-ebbaa580-4aa1-11eb-8534-285d724f6c93.png" width="900px">
  <img src = "https://user-images.githubusercontent.com/29679899/103373090-f83efe00-4aa1-11eb-9bd2-42b625dc9205.gif" width="500px">
</p>

<br/>

...sections 5 through 8 contained in depth analysis and illustrations of how attention-based bidirectional long-short term memory recurrent neural networks are constructed mathematically, how they work and their lay intutions... 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103373324-9d59d680-4aa2-11eb-9540-3153e5280bae.png" width="600px">
</p>

<br/>

...it detailed how to break down multi-classification problems into binary problems... 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103373214-5370f080-4aa2-11eb-9e78-e372fa9e9934.png" width="600px">
</p>

<br/>

...how to tune hyperparameters and make architectural changes to AB-BiLSTMRNNs... 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103373410-e14cdb80-4aa2-11eb-94a9-3fef668873e6.png" width="600px">
</p>

<br/>

...the mathematical/lay intuition regarding different loss functions, network layers, diagnosing bad network performance, interpretability of deep learning models, accuracy scores for the f1, precision and recall of the network, and the list goes on. The final classification results of the final model ended up being approximately 53%.

Included in section 8 was the performance and f1 score for a series of shallow machine learning algorithms that were trained and tested on the same data. Here are their scores:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103373597-5c15f680-4aa3-11eb-862d-f7c439c7c4af.png" width="290px">
</p>

<br/>

There's no hope of recovering the work and I regret not having a backup. Lesson learned. While this work was important to me, I will not try to recreate it, but I'm optimistic about where the project has gone from here and I can't wait to post about it.

<br/>

# 9. shallow algorithms 

In the context of machine learning, the essence of deep learning is associated with network architectures containing many layers and their ability to learn hierarchical distributed representations with little to no prior knowledge of the target problem. Traditional *shallow* machine learning algorithms usually lack a multitude of layers and require a large number of features to be engineered for them so they can learn the representations in the target problem (although feature engineering is not exclusive to shallow learning). The first shallow learners trained on our data are count vectorization and TF-IDF.

<br/>

# count vectorization & term frequency, inverse document frequency 
 
CountVectorizer is a bag-of-words (BOWs) model from the sklearn open-source library. A bag-of-words is a way of simplifying the representation of a given sentence by breaking it down into a token.
 
<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103373771-d3e42100-4aa3-11eb-9be0-1f0d17d082a5.jpg" width="300px">
</p>

<br/>
 
BOW models assign a weight to each tokenized word, analogous to the frequency in which it shows up in the document and corpora. In turn, this generates a document term sparse matrix (sparse matrices are mostly comprised of zeros). All columns being a document and each row being a token. 
