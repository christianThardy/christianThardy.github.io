# finding similarities between friends by way of social media using shallow and deep natural language processing

<br/>

# abstract

With the surge of social media, the internet has become an interesting and spirited domain in which billions of individuals from all around the world interact, share, post and conduct many daily activities. The immense size of social media data makes it notably different from classic data sources, and the mainly user generated data can be unbelievably noisy and unstructured. In social media mining, social media is considered a world of social atoms (i.e. *individuals*), and entities (e.g. *content, sites, networks etc.*) signaling interactions between social atoms and entities (et al. *Zafarani, Abbasi and Liu*). In this analysis I propose a way of looking at interactions governed solely between social atoms by collecting, mining and measuring the interactions between these social atoms to discover whatever salient patterns reside in each atom as they are projected through the lens of social media.

It is my goal to make sense of a Facebook Messenger chat group belonging to a group of friends that has been active for about 8 years and I hope to find relationships within each members text to derive an overall view of what those relationships mean to each other. The shared meaning within these conversations go back pretty far, and should be reinforced when you look over the entire body of conversational sentences. I do not use the word *meaning* as a way to represent strings of words relating to the intent of a speaker per say. The meaning I'm after in this analysis is derived from *form* rather than context of use. This is to say that we will learn which words are similar in place of commonsense reasoning without truly knowing what each atoms' private mental state infers. 

The overall text fragments within this corpus will be very short, considering the conversations held within the medium are stream of conscious by nature. I will explain in great detail, methods of deriving meaning from these short series of sentences using dense, distributed word embeddings by way of deep learning. After the text has been cleaned, each word will be mapped onto points in a high dimensional space to further reduce *meaningless* words to obtain *meaningful* words for each user. 

During the text extraction about 25,251 instances from the sum of each users name--originally ending at the 302,733rd instance--was lost. In an attempt to label this wealth of data so that it can be included in modeling user sentiment and topics, I propose a binary classification task to classify each users name to their respective text using an attention-based bidirectional long, short term memory recurrent neural network to learn the relevant features of each users text. The AB-BiLSTMRNN results will be weighted against engineered features tested on more classical information retrieval and shallow learning techniques such as term frequency inverse document frequency and sparse count vectorization trained with semantically weighted word vectors, logistic regression, gradient boosted trees, naive bayes and support vector machines.

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

In Katzian semantics--a form of generativist structural semantics--it is said that word meanings are defined in terms of their combination of simpler conceptual components, therefore word meanings are structured entities whose semantic markers reproduce the structure of the represented meaning and whose labels are the words conceptual components. Katzian semantics has its flaws[<a href="https://plato.stanford.edu/entries/word-meaning/#GenSem" title="Katzian semantics" rel="nofollow">'</a>], but the idea that words and their meanings can belong to a structure that lends itself to analysis is still very relevant.  

In an attempt to illustrate the idea of meaning, let's put the words `car`, `bus`, `road` and `driving` into a bag, mix them all up and dump them on the floor. Looking at the words, what do we see? The meaning between a group of words is equal to the distance between them, based on the likeness of their meaning[1]. I'm not just talking about words that appear similar to each other as opposed to just similarity--which can be estimated based on a set of rules, principles and processes that govern the structure of sentences in a given language--what I'm saying is that no matter what order the words are in, `car` will be similar to `bus` and both words are related to `road` and `driving`. This specific type of similarity is the building block of various mathematical tools that are used to estimate the strength of the semantic relationship between each unit of language.

Let's say you want to look at the way a person writes. You could very well end up asking questions like, *"What words are they using? How often are they using similar words?"*. Term frequency-inverse document frequency can be described as a method that looks at a persons vocabulary. Term frequency is how often you use particular words and inverse document frequency is how rare those words are across the document as a whole. The idea is that as people use certain words that aren't common, those are the words that are particularly strong signatures in that individuals writing style. So we're basically able to analyze that individuals' vocabulary. 

But this method is far from perfect and it ignores some of the most fundamental ideas of meaning. If you're just counting the frequency of words, we're missing out on why those words connect, which is pretty crude. The frequency of a word is somewhat misleading because you have to take into account how words appear together to form something meaningful. When we think of natural language processing as a way to potentially overcome this, we must also make computational linguistic goals. I've taken a sample sentence from one users corpus and used it as input to a linguistics parser built by the NLP Group at Stanford. This program is able to go through the sentence and establish a words syntactic role.

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

Generally, linguistics is centered around how sentences are basic units of thought, so we can conclude that if things start appearing together in a sentence a lot, we can take away meaningful structures from sentences. The parsed words in the sentence above belong to a corpus of extremely unstructured data, and while these are relatively unstructured data points, at the sentence and word level they are structured. This is exciting because we can begin to look at this text as being loosely structured sequences from which we can extract meaning. 

<br/>

# 1. collecting, and extracting the data

The text that I will be using came from Facebook.

<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/29679899/101291371-3feda500-37d6-11eb-900d-e9fc26c08faa.PNG" width="500px">
</p>

<br/>

After Facebook authorized the download, I was able to go through most of the data I've put on their platform over the years, but the only thing I'm interested in is the wealth of text that I've accumulated in this group:

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
# Dependencies

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
# Names of users 

d=csv.writer(open('bt_name_data_R.csv','w'))
d.writerow('Name')
data_name=soup.find_all('div',class_='_3-96 _2pio _2lek _2lel')

for data_name in data_name:
    names=data_name.contents
    d.writerow(names)
    
    
# Users dates & times 

d=csv.writer(open('bt_date_data_R.csv','w'))
d.writerow(['Date & Time'])
data_date_time=soup.find_all('div',class_='_3-94 _2lem')

for data_date_time in data_date_time:
    dates_times=data_date_time.contents
    d.writerow(dates_times)
    
    
# Users messages 

d=csv.writer(open('bt_message_data.csv','w'))
d.writerow(['Message'])
data_message=soup.find_all('div',class_ = '_3-96 _2let')

for data_message in data_message:
    messages=data_message.get('_3-96 _2let')
    d.writerow([messages])
```
<br/>

# 2. initial data exploration 

Datasets are like a good satirical bildungsroman[<a href="https://en.wikipedia.org/wiki/Bildungsroman" title="What is a bildungsroman" rel="nofollow">'</a>]. They will give you many ambiguous ideas, and sometimes they will even have an interesting story to tell. Exploring your data is a very important pillar of data analysis, because it gives a sense of what can be done with it and what may be possible. The first few reduction methods in this post will attempt to deconstruct the signal in our dataset into a set of features that will help our algorithms learn something meaningful. The only way you can handcraft good features for your data is by visualizing it, and after extracting the text from the html file and exporting everything into a nice tabular format, our first visualization is just a matter of several dependencies and less than 10 lines of code.

```python
# Dependencies

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
# Data

dataset = pd.read_csv('bt_fb_messenger_data.csv').fillna(' ')


# Shape of data

print("Training Data Shape: ", dataset.shape)
```
## Training Data Shape: (302731, 4)

<br/>

`pd.read_csv` is a function in `pandas` that allows data to pass into python, but unlike python's native `.read()` function, we can perform containerization operations that are exclusive and specific to the pandas library. `pd.read_csv` is contained within the dataset variable which is then used to determine the shape of the data using the .shape function. After executing these two lines of code, a `pandas` matrix is returned. A matrix in pandas is synonymous with arrays in python. Arrays are data structures that can hold any rendition of data in a structured way, and a matrix is simply a two-dimensional data structure where numbers are arranged into rows and columns. The returned integer tells us that there are 302,731 rows, within the table and 4 columns in a 2-dimensional state. 

In order to understand the text, we'll first need to get a sense for how the variables are distributed. Distributions show us the possible values for variables and how often they occur. Maybe we want to compare the distribution of a variable across levels of other variables. Are we working with univariate, bivariate or multivariate distributions? You can easily make some of these assumptions by just looking at the data in its raw form, but visualizing it helps us to understand it on a more intuitive level. 

Since each users response in the app is dictated by the amount of times they send a message, lets obtain the frequency of each users response. We'll isolate the feature `Names` from the dataset in the names variable to determinehow many total occurrences there are for each user. To visualize the frequency, we'll specify the plots height and width with the figsize element and call it from the `plt.figure` function.

`sns.barplot` supports the input data from `.index`, which specifies where the index values begin, and we use this to access the `pandas` data frame within the names variable, and `.values` allows the index values to be displayed. alpha specifies a semi-translucent tone to the bar plot and color simply returns a color for the bars. `plt.xlabel` and plt.ylabel applies a label to the `x/y` axis and `plt.show()` allows us to visualize the 6 lines of code.

```python
# Number of occurances for each user

names = dataset['Name'].value_counts()
plt.figure(figsize=(12,4)) 
sns.barplot(names.index, names.values,alpha=0.8,color=color[1])
plt.ylabel('Number of Occurrences',fontsize=12)
plt.xlabel('User Name',fontsize=12)
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
# Sample: bt_4

bt_4 = dataset['Name'] == 'bt_4'


# Shape of data

print('bt_4 data shape: ', bt_4.shape)
```

## `bt_4 data shape:  (38954, 4)`

<br/>

For this initial text preprocessing phase I'll use the spacy (NLP python library) pipeline feature to process 500 samples of `bt_4`'s text at a time to grab the sample word vectors and format them into numpy arrays. 

The first part of the code is used to clean the text by lemmatizing the words and removing personal pronouns,--in `spacy` the lemmatized string of personal pronouns is `'-PRON-'` --stop words and punctuations. We'll dive deeper into the granular intricacies of text preprocessing later on, so for the sake of brevity let's dive into the code. 

<br/>

```python
# Dependencies

import spacy
nlp=spacy.load('en_core_web_sm')
punctuations=string.punctuation


# Create function to clean up text by removing personal pronouns, stopwords and punctuation

def cleanup_text(docs,logging=False):
    texts=[]
    counter=1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter,len(docs)))
        counter += 1
        doc=nlp(doc,disable=['parser','ner'])
        tokens=[tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens=[tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens=' '.join(tokens)
        tokens=re.sub("(^|\W)\d+($|\W)", " ",tokens)
        tokens=re.sub('[^A-Za-z0-9]+', '',tokens)
        texts.append(tokens)
    return pd.Series(texts)
```

<br/>

The next few lines of code will obtain all the words from `bt_4`'s message feature and put them in a list. Then the text will be cleaned using the `clean_text` function, which will remove common stop words, punctuation and make all words lowercase. Next, all of the 's will be removed because `spacy` doesn't remove this contraction when lemmatizing words. Lastly the count occurrences of all words are gathered.

<br/>

```python
# Collect all text associated to bt_4

bt_4_text=[text for text in dataset[dataset['Name'] == 'bt_4']['Message']]


# Clean bt_4 text

bt_4_clean=cleanup_text(bt_4_text)
bt_4_clean=' '.join(bt_4_clean).split()


# Remove words with 's

bt_4_clean=[word for word in bt_4_clean if word != '\'s']


# Count all unique words

bt_4_counts=Counter(bt_4_clean)
bt_4_common_words=[word[0] for word in bt_4_counts.most_common(30)]
bt_4_common_counts=[word[1] for word in bt_4_counts.most_common(30)]
```

<br/>

After the text has been preprocessed, the 30 most frequently occurring words for `bt_4` are visualized using `matplotlib` and `seaborn`.

<br/>

```python
# Plot 30 most commonly occuring words

plt.figure(figsize=(20,12))
sns.barplot(x=bt_4_common_words,y=bt_4_common_counts)
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
stopwords=stopwords.words('english') 


# Number of words in the text 

dataset["num_words"]=dataset["Message"].apply(lambda x: len(str(x).split()))


# Number of unique words in the text 

dataset["num_unique_words"]=dataset["Message"].apply(lambda x: len(set(str(x).split())))


# Number of characters in the text 

dataset["num_chars"]=dataset["Message"].apply(lambda x: len(str(x)))


# Number of stopwords in the text 

dataset["num_stopwords"]=dataset["Message"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))


# Number of punctuations in the text 

dataset["num_punctuations"]=dataset['Message'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))


# Number of upper case words in the text 

dataset["num_words_upper"]=dataset["Message"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))


# Number of title case words in the text 

dataset["num_words_title"]=dataset["Message"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))


# Average length of the words in the text 

dataset["mean_word_len"]=dataset["Message"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# Truncated violin plot of the number of words by user

dataset['num_words'].loc[dataset['num_words']>80]=80
plt.figure(figsize=(12,8))
sns.pointplot(x='Name',y = 'num_words',data=dataset)
plt.xlabel('User',fontsize = 20)
plt.ylabel('Number of words in text',fontsize=15)
plt.title('Number of words by User',fontsize=20)
plt.show()
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
# Dependencies

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
corpus=[]
for i in range(0,38954):
    clean_text=re.sub('[^a-zA-Z]', ' ', str(bt_4_text[i]))
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
clean_text=[ps.stem(word) for word in clean_text if not word in set(stopwords.words('english'))]
clean_text=[word for word in clean_text if not word in set(get_stop_words('english'))]
clean_text=[word for word in clean_text if word not in STOPWORDS]
clean_text=[word for word in clean_text if word not in bt_4_additional_stopwords]
```
## `['goddamn', 'season', 'queer', 'eye']`

<br/>

With our example illustrated, now we can put everything into a nice `for` loop that will iterate over every document in the corpus. 

<br/>

```python
corpus=[]
for i in range(0,38954):
    clean_text=re.sub('[^a-zA-Z]', ' ', str(bt_4_text[i]))
    clean_text=clean_text.lower()
    clean_text=clean_text.split()

    # text stemming & stop word removal
    ps=PorterStemmer() 
    clean_text=[ps.stem(word) for word in clean_text if not word in set(stopwords.words('english'))]
    clean_text=[word for word in clean_text if not word in set(get_stop_words('english'))]
    clean_text=[word for word in clean_text if word not in STOPWORDS]
    clean_text=[word for word in clean_text if word not in bt_4_additional_stopwords]
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

bt4Vec=Word2Vec(sentences=corpus,size=100,window=5,min_count=4, 
                workers=8,sg=0,iter=30,alpha=0.020)
bt4Vec=bt4Vec.wv
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
## `bt_4 corpus length: 31413`

Each token in the corpus can be thought of as an array of text, with each discrete symbol corresponding to an embedding vector inside the model. The embeddings are contained by one-dimensional vectors, with each vector corresponding to a word in the vocabulary. Each word is mapped to vectors of real numbers, so that's 31,413 tokens and each one will contain several of their own embeddings and is expressed as the sequence in which the word occurs, but as a vector. 
 
To visualize the one-dimensional embeddings, we'll need to transform them into two-dimensional tensors. A tensor is a variable that has `n` indices where each index covers a range of dimensions of the three-dimensional space. In short, they are abstractions of scalars that give us a good framework to represent our one-dimensional vectors in Euclidean space.

<br/>

# restatement

I started this version of the project, so all of the work in this post, in August of 2017. It was completed in April of 2018 and was posted to my old blog the following year. The platform I was using at the time to host my content had many problems and in November of 2020 I lost the next 5 sections of the project. As mentioned in the introduction, the next sections would have included working with deep learning based algorithms to classify each member of the dataset to their respective text where the label was lost. 

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

...the mathematical/lay intuition regarding different loss functions, network layers, GloVe embeddings vs word2vec embeddings, diagnosing bad network performance, interpretability of deep learning models, accuracy scores for the f1, precision and recall of the network, and the list goes on. 

There's no hope of recovering the work and I regret not having a backup. Lesson learned. While this work was important to me, I will not try to recreate it, but I'm optimistic about where the project has gone from here and I can't wait to post about it.

<br/>

# 9. shallow algorithms 

In the context of machine learning, the essence of deep learning is associated with network architectures containing many layers and their ability to learn hierarchical distributed representations with little to no prior knowledge of the target problem. Traditional *shallow* machine learning algorithms usually lack a multitude of layers and require a large number of features to be engineered for them so they can learn the representations in the target problem (although feature engineering is not exclusive to shallow learning). The first shallow learners trained on our data are count vectorization and TF-IDF.

<br/>

# sparse count vectorization & term frequency, inverse document frequency 
 
`CountVectorizer` is a bag-of-words (BOWs) model from the sklearn open-source library. A bag-of-words is a way of simplifying the representation of a given sentence by breaking it down into a token.
 
<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103373771-d3e42100-4aa3-11eb-9be0-1f0d17d082a5.jpg" width="300px">
</p>

<br/>
 
BOW models assign a weight to each tokenized word, analogous to the frequency in which it shows up in the document and corpora. In turn, this generates a document term sparse matrix (sparse matrices are mostly comprised of zeros). All columns being a document and each row being a token. 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103449145-f62b9980-4c71-11eb-91e9-ff79e633fa0f.gif" width="500px">
</p>

<br/>

The idea behind BOWs is that you can take your document and break down each sentence into tokens (either unigrams, bigrams or trigrams) and you throw them into the model. It's like you're cutting up a piece of paper, writing a single word on each piece and you dump those into a bag. It's very unsophisticated, and the order is loosely structured. It's not like you're working your way through a model from left to right, maintaining grammar or even sequential probability.

When we use BOWs, what we're looking for is overlap. Whatever word we're looking for to complete a compound word or a sentence, does it have a trigger word in the bag, and if the trigger words are there, more than likely the correct document can be found, which will be the goal for the classification task. So we can begin to ask, *"Does this corpus have certain trigger words in the bag?"*. It doesn't matter what order they appear, it doesn't matter what grammatical path we pave through the sentence, all that matters is that those words are present. 
 
Out of the two methods covered in this section, `CountVectorizer` is the most basic. It allows you to build a lexicon of familiar words and encode new documents using that vocabulary by counting the number of times a token shows up in the document and it uses this value as its weight. So `CountVectorizer` is primarily based on the count of words.

`TfidfVectorizer` (also from sklearn) is different from `CountVectorizer` in that it allows us to calculate word frequencies. So the weight given to every token relies on its frequency in a document and how that term is reoccurring in the corpora[28]. `TfidfVectorizer` is based on the TF-IDF (term frequency, inverse document frequency) method and is the information retrieval technique that made the search engine likes of Google possible.
 
So term frequency measures how often a word occurs in corpora. Since most documents vary in range, it is possible that a term would appear many times in long documents rather than in shorter ones. Therefore, the word frequency is divided by the document's range as a way of normalization. To achieve word (term) frequency you will compute:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103449204-fe380900-4c72-11eb-8834-ff471cae6aba.png" width="600px">
</p>

<br/>

Inverse document frequency estimates how significant a word is and when calculating term frequency on a BOWs, all words are deemed equally significant. Although, certain words like `is`, `of` and `that` will show up a lot, and have little importance. Therefore, the frequent words need to be weighed ↓ while the rare ones are scaled ↑ by way of a logarithm. This is done by computing the following:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103449223-51aa5700-4c73-11eb-8761-c1a1dd05e75c.png" width="600px">
</p>

<br/>

A bag-of-words model takes a document and transforms it into a vector. The vector size or the width of the vector, is going to be the number of columns in your feature space with one vector for every column. The gif below illustrates this document to vector transformation

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103449247-a8b02c00-4c73-11eb-8833-2b4b2507c07b.gif" width="600px">
</p>

<br/>

By formulating the problem like this you can ask, *"Is a given word present in the document?"* which will return a `0` or a `1`, a `yes` or a `no`. So if the word is present in this document then we put a `1` in the column above and if not we put a `0`.

This is called a sparse vector because there is a sparsity of `1`'s, and the rest of the numbers are mostly `0`'s. The opposite of this is a dense vector which is squished down to many real numbers between `0` and `1`. An example of this can be seen at the end of section 4.
 
So we're left with one vector per document, many documents in the corpus and however many rows. If you're looking for the word pet food in the corpus, the algorithm will look through all the documents for documents where the pet column is `1` and the food column is `1`.
 
Any documents that don't have both columns set to `1` are discarded which allows you to sort the documents in order of relevance to the word (or user). Instead of storing a `1` or a `0` in the column for a word for every document, lets store a tally of the number of times that word appears in the document. Instead of storing a `1` for pet lets store the number of times pet appears in the corpus. Now we can sort the documents when searching for pet food retrieved by the number of occurrences of that word in the document. More pets and foods in the document means more relevance to the query.

Sticking with the Google example, people abused this system a long time ago by way of a very old black hat search engine optimization technique called keyword stuffing. 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103449367-56700a80-4c75-11eb-810b-4c0050fefa64.jpg" width="500px">
</p>

<br/>

When Google had this naive version of TF-IDF implemented on their search engine, people could add to the end of any web page, any keywords they wanted to show up for as highly relevant for search queries. So way back when, it would have been possible to add `cat cat cat cat cat cat` and `food food food food food food` as many times as you want on your website, with white font on white background, font size 0. Given all of the "algorithm updates" its safe to say that Google's version of information retrieval has been pretty abstracted from this simple example.
 
The way Google was able to abstract away from this naive example was by designing millions of handcrafted features, conceptually like the handcrafted features mentioned near the beginning of this analysis. They fancied the word signal rather than feature, but their researchers would improve Google search by coming up with some new features, add them as new features for the algorithm, and their search engine would get better. But this was about until 2015. As of fall of 2019 they're most likely only relying on BERT, which is a deep learning model that learns a great deal of these features on its own, and being a large language model it is focused on the intent and semantics of content. This makes gamification of the search engine via SEO virtually impossible.
 
To summarize, term frequency-inverse document frequency is a weighted numerical estimate used to calculate how essential a word is to a body of text. The significance of a given word increases relatively to the number of times a word appears in a document but is offset by the frequency of the word in the large body of text. TF-IDF also shines when computing document relevance[29], and trying to figure out how similar documents are to each other.

In the code section below, we will train TF-IDF and count weighted vectors on the part of speech features of `bt_1` and `bt_5` using 6 shallow algorithms and 2 versions of `ExtraTreeClassifier`'s trained on TF-IDF/count weighted GloVe embeddings. Our objective metric to measure performance is based on the weighted F1 metric, which is a weighted average of the relative equal contribution of recall and precision. So:

<br/>

## F1 = 2 * (precision * recall) / (precision + recall)

<br/>
 
The dataset is passed to the `data_str` → `word_tokenize` → `tokenized_text` variables which tokenize the dataset. The final `tokenized_text` variable is passed to the part-of-speech tagger `nltk.pos_tag` as the `list_of_tagged_words` variable, which is then passed to a very special function. The set function is reliant on set theory. A set is a defined collection of easily distinguishable objects or in our case tokens. 

```python
# Dependencies

from tqdm import tqdm
from nltk import word_tokenize
from sklearn import preprocessing
from nltk.tag import pos_tag_sents
from sklearn.model_selection import train_test_split


# Data

dataset=pd.read_csv('bt_data_train_set_1_5.csv').fillna(' ')


# Transforms target variable into 0s and 1s for classification

lbl_enc=preprocessing.LabelEncoder()
y=lbl_enc.fit_transform(dataset.Name.values)


# Returns every row as a string inside of a list 

data_str=''
for i in dataset.itertuples():
    data_str=data_str + str(i.Message)
    
    
# Tokenizes text

tokenized_text=word_tokenize(data_str)


# Appends list as a function to retrieve 
# NLTK part of speech tags

list_of_tagged_words=nltk.pos_tag(tokenized_text) 
```

<br/>

Placing the tagged tokens in a set object corresponding to each word and its part-of-speech, the set will separate the tokens into a number of categories to reduce the number of operations needed to check if a particular token is in the set.

```python
'''Based on hash-tables, which are continuous vectors 
   similar to python dictionaries, set_pos transforms 
   list_of_tagged_words into a highly optimized,
   iterable method that will make sure pos_tags is 
   contained within the object its called. We only 
   want the features in pos_tags included in the 
   final version of list_of_tagged_words before we 
   split the train and test sets'''

pos_set = (set(list_of_tagged_words))

'''Specifies the parts of speech
   we want to capture and groups 
   them together''' 

pos_tags = ['PRP','PRP$', 'WP', 
            'WP$','JJ','JJR','VB', 
            'VBD','VBG', 'VBN','VBP', 
            'VBZ','JJS','EX','IN','CD',
            'CC','NN','NNS','NNP','NNPS']
```

<br/>

We're doing this so that when the set variable is called within `list_of_words`... 

```python
# Removes the 1st index of set object

list_of_words = set(map(lambda tuple_2: tuple_2[0], filter(lambda tuple_2: tuple_2[1] in pos_tags, pos_set)))
```
<br/>

...it can help retrieve the parts of speech that are identified in the `pos_tags` variable and only retrieve the items in that variable. To break down `list_of_words` linearly, set allows us to group each function in the variable intelligently → `map` and `lambda` allow our list of inputs, `pos_tags` and `set_pos` to be passed through each function one by one in the variable → they also allow the list of functions,` tuple_2: tuple_2[0]`, do the same and represent our word whose index is `0` (wipe).

Breaking down `tuple_2: tuple_2[0]` a bit, they're packing each word from `set_pos`, while `filter(lambda tuple_2: tuple_2[1]` unpacks and discards the words part of speech tags indexed by `1` `(NNS)` that are not specified in the `pos_tags` variable. `filter` is the function that handles this by only returning the parts of speech specified in `pos_tags`. Using nested sets in combination with tuples allows iterating over the massive rows of text to happen very quickly. 

`dataset['pos_features']`... 

```python
# Transforms bt_1 & bt_5 Message vectors 
# Based on functions from list_of_words

dataset['pos_features'] = dataset['Message'].apply(lambda x: str([w for w in str(x).split() if w in list_of_words]))
```

<br/>

...takes the `'Message'` column from the dataset and applies the `list_of_words` variable which transforms `bt_1` and `bt_5`'s sentences into individual words that can be categorized as personal possessive pronouns, nouns, verbs, adjectives, existential phrases, prepositional phrases, coordinating conjunctions, and cardinal digits that are in each row of each users respective document. 

Now we have a new dataset that only contains the specified part of speech features. Next, we'll split them into the train and validation sets, both of which will be stratified and shuffled.

```python
# Split data into xtrain/ytrain xval/yval sets

xtrain,xval,ytrain,yval = train_test_split(dataset.Message.values,y, 
                                           stratify=y, 
                                           random_state=42, 
                                           test_size=0.1, shuffle=True)
```

<br/>

We'll reappropriate the script used during construction of the AB-BiLSTMRNN to import the the `glove_vectors`... 

```python
# Import glove embeddings

glove_vectors={}
e=open('glove.840B.300d.txt') # Need the full representation which includes stopwords

for p in tqdm(e):
    real_num=p.split(' ')
    word=real_num[0]
    coefs=np.asarray(real_num[1:],dtype='float32')
    glove_vectors[word]=coefs
e.close()

print('Found %s word vectors.' % len(glove_vectors))
```
## 2196018it [02:35, 14077.23it/s]
## Found 2196017 word vectors

<br/>

...and we'll use them to create TF-IDF, count-vectorized weighted GloVe vectors which will serve as input for the `ExtraTreesClassifer`.

In order to train each model at the same time we need to construct something called a `Pipeline`. Ten of them to be exact: 

```python
# Dependencies

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier 
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


'''Multinomial/bernoulli naive bayes, logistic regression 
   xgboost and a support vector classifier
   all with countvec and tfidf weighted features.'''

multi_nb = Pipeline([('count_vectorizer',
                       CountVectorizer(analyzer=lambda x: x,token_pattern=r'\w{1,}',ngram_range=(1,3),
                                       stop_words='english')),('multinomial nb',MultinomialNB())])
                                      
multi_nb_tfidf = Pipeline([('tfidf_vectorizer',
                             TfidfVectorizer(analyzer=lambda x: x,min_df=3,max_features=None,
                                             strip_accents='unicode',token_pattern=r'\w{1,}',
                                             ngram_range=(1,3),use_idf=1,smooth_idf=1,sublinear_tf=1,
                                             stop_words='english')),('multinomial nb',MultinomialNB())])
                                            
bern_nb = Pipeline([('count_vectorizer',
                      CountVectorizer(analyzer=lambda x: x,token_pattern=r'\w{1,}',ngram_range=(1,3),
                                      stop_words='english')),('bernoulli nb',BernoulliNB())])
                                     
bern_nb_tfidf = Pipeline([('tfidf_vectorizer',
                            TfidfVectorizer(analyzer=lambda x: x,min_df=3,max_features=None,
                                            strip_accents='unicode',token_pattern=r'\w{1,}',
                                            ngram_range=(1,3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                                            stop_words='english')),('bernoulli nb',BernoulliNB())])
                                           
log_reg = Pipeline([('count_vectorizer',
                      CountVectorizer(analyzer=lambda x: x,token_pattern=r'\w{1,}',
                                      ngram_range=(1,3),stop_words='english')),
                                     ('logistic regression', LogisticRegression(C=1.0))])
                    
log_reg_tfidf = Pipeline([('tfidf_vectorizer',
                            TfidfVectorizer(analyzer=lambda x: x,min_df=3,max_features=None,
                                            strip_accents='unicode',token_pattern=r'\w{1,}',
                                            ngram_range=(1,3),use_idf=1,smooth_idf=1,sublinear_tf=1,
                                            stop_words='english')),('logistic regression',LogisticRegression(C=1.0))])
                                           
xgb = Pipeline([('count_vectorizer', 
                  CountVectorizer(analyzer=lambda x: x,token_pattern=r'\w{1,}',
                                  ngram_range=(1,3),stop_words='english')),('xg boost',
                                                                             XGBClassifier(max_depth=7,
                                                                                           n_estimators=200,
                                                                                           colsample_bytree=0.8,
                                                                                           subsample=0.8,nthread=10,
                                                                                           learning_rate=0.1))])
xgb_tfidf = Pipeline([('tfidf_vectorizer',
                        TfidfVectorizer(analyzer=lambda x: x,min_df=3,  max_features=None,
                                        strip_accents='unicode',token_pattern=r'\w{1,}',
                                        ngram_range=(1,3),use_idf=1,smooth_idf=1,sublinear_tf=1,
                                        stop_words='english')),('xg boost',
                                                                XGBClassifier(max_depth=7,
                                                                              n_estimators=200,
                                                                              colsample_bytree=0.8,
                                                                              subsample=0.8,nthread=10,
                                                                              learning_rate=0.1))])
svc = Pipeline([('count_vectorizer', 
                  CountVectorizer(analyzer=lambda x: x,token_pattern=r'\w{1,}',
                                  ngram_range=(1, 3), stop_words='english')),('linear svc',
                                                                               SVC(kernel='linear'))])
                                                                             
svc_tfidf = Pipeline([('tfidf_vectorizer', TfidfVectorizer(analyzer=lambda x: x,min_df=3,
                                                           max_features=None,strip_accents='unicode',
                                                           token_pattern=r'\w{1,}',ngram_range=(1, 3),
                                                           use_idf=1,smooth_idf=1,sublinear_tf=1,
                                                           stop_words='english')), ('linear svc', 
                                                                                     SVC(kernel='linear'))])
```

<br/>

`Pipeline`'s allow us to gather multiple steps or perform sequences of different transformations, that can be cross validated together while also allowing us to test a number of different algorithms and parameters. Defining each model is pretty simple and so I will only explain what's happening with the multi-nomial naive bayes algorithm:

```python
multi_nb = Pipeline([('count_vectorizer',
                       CountVectorizer(analyzer=lambda x: x,token_pattern=r'\w{1,}',ngram_range=(1, 3),
                                       stop_words='english')),('multinomial nb',MultinomialNB())])
                                      
multi_nb_tfidf = Pipeline([('tfidf_vectorizer',
                             TfidfVectorizer(analyzer=lambda x: x,min_df = 3,max_features=None,
                                             strip_accents='unicode',token_pattern=r'\w{1,}',
                                             ngram_range=(1,3),use_idf=1,smooth_idf=1,sublinear_tf=1,
                                             stop_words='english')),('multinomial nb',MultinomialNB())])
```

<br/>

Inside of the `multi_nb` and `multi_nb_tfidf` variables, an sklearn `Pipeline` function is defined with parameters specific to each vectorizer type. 

The `analyzer` in `CountVectorizer` needs to transform strings of words into features. We've previously taken ngram features from NLTK as input, so to vectorize an iterable of strings for `CountVectorizer` we'll use `lambda x: x` to express an anonymous function which will serve as the `analyzer`'s input. Using a `lambda` here is great because it effectively allows us to declare a simple inline function on a parameter that we don't actually need but requires an argument. `Token_pattern` defines what constitutes a token. `ngram_range` tells `CountVectorizer` the upper and lower boundries of ngrams to extract from the data.

N-grams are very important. After defining what a token is, traditionally you then split those tokens into a gram of some sort. n being the number of grams. If you're splitting the document into 1 grams or unigrams, you're splitting the document into its individual tokens. For example, `"How's" "the" "weather"` would be 3 tokens or 3 unigrams.

The document can also be split into 2 grams or bi-grams, `n` being 2-grams. `"How's the" "the weather"` would then be 2 bi-grams. Notice how the first bi-gram overlaps with the second bi-gram. This is done because the number of grams can increase the accuracy of the language model at the expense of compute power. Our upper bound and lower bound will be set to 1 and 3 for each algorithm. `'english'` is passed to `stop_words` and finally `'multinomial nb'` variable is initialized by `MultinomialNB()`.

`tfidf_vectorizer` shares similarities to `count_vectorizer`, noteably the `analyzer`, `token_pattern`, `ngram_range` and `stop_words` parameters.

Where they differ is `min_df`, ignores words with a document frequency lower than `3` → `max_features` set to None ignores frequent/rare terms and considers the entire corpus during the term frequency, inverse document frequency transformation → `strip_accents` normalizes the characters in each ngram set by removing accents and `'unicode'` removes special characters thus reducing the dimensionality of the text → `use_idf` is set to `1` = True so the inverse document frequency is considered during transformation → `smooth_idf` smoothes the inverse document frequency weights called from `user_idf` in a way that if a word in the corpus was never seen by the training data but occurs in the test set, it allows that word to be processed → `sublinear_tf` computes the logarithm of the frequency and scales words logarithmically so that rare words carry as much significance as words that might be common but have many occurrences.

The algorithms used in this code section are Multinomial & Bernoulli Naive Bayes, logistic regression, XGBoost and a support vector classifier[30]. Later on I will only explain the algorithms that return the most favorable results on our weighted F1 metric.

In order to use GloVe weighted TF-IDF and count vectorizer embeddings on the `ExtraTreeClassifier`, we'll need to tap into some syntactic sugar['] and define custom TF-IDF and count vectorizer classes and methods.

`CountVectorizerEmbeddings`... 

```python
# Dependencies

from collections import defaultdict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score


'''Vectorizes the text by taking the mean 
   of all the  vectors corresponding to individual 
   words in a given vector mapping''' 

class CountVectorizerEmbeddings(object):
    def __init__(self, glove):
        self.glove = glove
        if len(glove)>0:
            self.dim=len(glove[next(iter(glove_vectors))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.glove[w] for w in words if w in self.glove] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
```

<br/>

...is defined as a new type of class object which will allow instances of its type to be passed on to `ExtraTreesClassifier`. As explained in the Attention section of the 7th part of this post, `__init__` , which is our constructor, allows class objects to accept arguments and will be the main definer of the `CountVectorizerEmbeddings` class where self assigns the glove vectors as static instances of the class object `CountVectorizerEmbeddings`.

The following `if` / `else` statement says that if the length of the `glove_vectors` are greater than 0, the instance dim of the class object will ask next (ran inline for conveniece) to process the contents of `glove_vectors` and will iterate over the length of the vectors one time. Otherwise (`else`) if the length of the `glove_vectors` are less than 0, `dim` will be equal to 0.

The `fit` function allows the class instance to define our `X` and `y` variables that will eventually contain `xtrain` and `ytrain`. This is where the model will fit the parameters from our initial class to the data, which will enable the model to learn.

Finally, the `transform` function applies the parameters obtained by the `fit` function. It will transform documents into a document-term matrix, and extract token counts using the argument specified by fit or the class instance provided by the initial class. It will do this by returning a numpy array, `np.array([])`.

We need the `glove_vectors` inside of an array as they are the input type that machine learning algorithms like, because numerical representations are what they can understand. Subsequently, it will also transform  `xtrain` into an array when passed as its argument.

Calling `transform` will return the `np.array([])` which contains the np.mean object (literally takes the arithmetic mean of its arguments) which contains an expression called a nested list comprehension as its argument, that will retrieve each word contained in `glove_vectors` and will append them to `xtrain`. `self.glove[w]` to represent each row of words in our `glove_vectors` as `w`.
 
The conditional states, for the `glove_vectors` retrieved in all words, if the `glove_vectors` are located in the `self.glove` instance or a matrix of zeros defined by the `self.dim` instance from the initial class, and if the words in `X` are evaluated as `True` with respect to `self.glove`, the `transform` function can initialize and the `CountVectorizerEmbeddings` class to be used in conjunction with any classification algorithm of our choice.

The `__init__` function of the `TfidfVectorizerEmbeddings` class... 

```python
'''Vectorizes the text by taking the mean 
   of all the  vectors corresponding to individual 
   words in a given vector mapping''' 

class TfidfVectorizerEmbeddings(object):
    def __init__(self, glove):
        self.glove = glove
        self.word2weight = None
        if len(glove)>0:
            self.dim=len(glove[next(iter(glove_vectors))])
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        '''If a word was never seen - it must be at least as infrequent
           as any of the known words - so the default idf is the max of 
           known idf's'''
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]) 
    
        return self
    
    def transform(self,X):
        return np.array([
                np.mean([self.glove[w] * self.word2weight[w]
                         for w in words if w in self.glove] or
                        [np.zeros(self.dim)],axis=0)
                for words in X
            ])
```

<br/>

...follows the same conventions as `CountVectorizerEmbeddings` but the `fit` and `transform` functions are a bit different. Similar to `CountVectorizerEmbeddings` `fit` function, `X` and `y` are defined. The functionality from sklearn's `TfidfVectorizer` is borrowed out of convenience to define `max_idf` so that if a word was never seen, it needs to be as rare as any of the known words so that the baseline `idf` is the max of known `idf`'s.

So `max_idf` is defined by the python function max which takes `tfidf.idf_` as its argument. `.idf_` is an attribute of the `TfidfVectorizer` method that takes the TF-IDF score of each feature and makes it retrievable. When `max_idf` is computed,  its passed to the `self.word2weight` variable as an argument of the `defaultdict`.

`defaultdict` will create a dictionary that counts the number of times each word (key) occurs in each users documents and count the number of times the top occurrences of words across all documents in each users corpora best represents the total corpus (TF-IDF, GloVe weight value). `defaultdict` receives `lambda: max_idf` as its default value, and allows the dictionary to return when its trying to be retrieved by a hypothetical key.

Keys that do exist are represented by a list comprehension of key-value pairs (w representing our words and `tfidf.idf_[i]` denoting a one-dimensional index of all TF-IDF, GloVe weighted scores in our corpus) that are tossed inside of `tfidf.vocabulary_`, which determines what a key is and what a weight is, and `.items()` iterates over the key-value pairs which will automatically assign the first variable as the key (word) and the second variable as the value (TF-IDF, GloVe weighted value) for the key. This is necessary so that words from our users corpora can attain their own individual weighted TF-IDF and GloVe features.

Like `CountVectorizerEmbeddings`, `TfidfVectorizerEmbeddings` will transform the parameters defined by the `__init__` and the fit function, but in a very different way. np.mean not only takes a nested list comprehension to retrieve each word in `glove_vectors`, but it also calculates the weighted sum of TF-IDF and GloVe embedding vectors;                   

<br/>

## `self.glove[w]  *  self.word2weight[w]`

<br/>

Like `CountVectorizerEmbeddings` `transform` function, if the elements within the nested, iterable list is `True`, the function will place the weighted embeddings into a sparse array of zeros with the same length as `self.dim`.
 
`CountVectorizerEmbeddings` and `TfidfVectorizerEmbeddings` are essentially bag-of-words representations. Where they differ is that the former is simply based on the number of occurrences of words, whereas the latter determines which words in the corpora are favorable based on each words document frequency denoted by the total number of words in the corpora.
 
When we weight each measure by the semantic information the GloVe vectors capture, this will further maximize our classification algorithms ability to generalize, because as effective as count vectorization and TF-IDF are on their own, adding weights to each word based on its frequency within the GloVe embeddings will allow count vectorization and TF-IDF to also take semantic similarities into account during classification.
  
The last two sections were pretty dense, but things will get a lot easier from here. Taking queues from notebook 43, we'll assemble an sklearn `Pipeline` to encompass the `ExtraTreesClassifier`'s, which will each take `n_estimators` that are equal to `200`...

```python
'''Glove vectors passing through a stack of random decision trees
   that will be trained using tf-idf weighted + glove weighted vectors'''

stacked_r_tree_glove_vectors=Pipeline([(
     'glove vectorizer',CountVectorizerEmbeddings(glove_vectors)),
    ('stacked trees',ExtraTreesClassifier(n_estimators=200))])

stacked_r_tree_glove_vectors_tfidf=Pipeline([(
     'glove vectorizer',TfidfVectorizerEmbeddings(glove_vectors)),
    ('stacked trees',ExtraTreesClassifier(n_estimators=200))])
```

<br/>

...which are the number of trees in the classifier... 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104081535-4b7d2300-51fd-11eb-8572-6f074307e56b.png" width="400px">
</p>

<br/>

...and our embedding classes.

Each model that we're training is stored in `all_models`: 

```python
# Places algorithm variables in a neat tabulated format

from tabulate import tabulate
# All 6 models
all_models=[('multi_nb',multi_nb),
            ('multi_nb_tfidf',multi_nb_tfidf),
            ('bern_nb',bern_nb),
            ('bern_nb_tfidf',bern_nb_tfidf),
            ('log_reg',log_reg),
            ('log_reg_tfidf',log_reg_tfidf),
            ('xgb',xgb),
            ('xgb_tfidf',xgb_tfidf),
            ('svc',svc),
            ('svc_tfidf',svc_tfidf),
            ('glove_vectors',stacked_r_tree_glove_vectors),
            ('glove_vectors_tfidf',stacked_r_tree_glove_vectors_tfidf)]


# Takes average of each algorithms output via the weighted f1 evaluation metric

disordered_scores=[(name,cross_val_score(model,xtrain,ytrain,
                                         scoring='f1_weighted',
                                         cv=2).mean()) for name,model in all_models]


# Sorts and prints the evaluation score of each algorithm

scores=sorted(disordered_scores,key=lambda x: -x[1])
print(tabulate(scores,floatfmt='.4f',headers=('model','score')))
```
<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/103373597-5c15f680-4aa3-11eb-862d-f7c439c7c4af.png" width="290px">
</p>

<br/>

The baseline score obtained from the AB-BiLSTMRNN was 53%. Each algorithm from this section is evaluated on the same data and scores moderately to much higher than the AB-BiLSTMRNN. The weighted GloVe, TF-IDF embeddings trained on the `ExtraTreeClassifier` scored 70%. The vanilla XGBoost algorithm trained on the vanilla count vectorizer model scored 69% and the TF-IDF weighted Bernoulli Naive Bayes algorithm scored 68%. These scores were obtained only using the part of speech features. Its probable the score would increase if we included the numerical count based features. Relying on an end-to-end, deep learning architecture to independently learn the correlating features in a dataset is not always the best path to take. For this problem, feature engineering is the best tool to use.
 
In the next section I will explain the defining aspects of the gradient boosted algorithm extremely randomized trees, XGBoost and the probabilistic algorithm Bernoulli Naive Bayes in a little more depth, and interpret their decision making processes.

<br/>

# boosting
 
Boosting algorithms are based on decision trees and random forests. They make their predictions by searching for the best point at which the algorithm divides the training data at the root into multiple small-scale pieces of data (leaves) based on correlated variables in the data and eventually the split with the greatest advancement is chosen as the strongest predictor.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104081766-c98df980-51fe-11eb-88c8-cabcf70ae69f.png" width="500px">
</p>

<br/>

While gradient boosting and neural networks can be very different, they share one very important component. Training a gradient boosting machine is essentially performing gradient descent by moving some parameter `x` in vector space, while minimizing a loss function `L`, that compares the target `y` to a nonlinear approximation `ŷ`.

Gradient boosting machines are made up of multiple weak, but computationally efficient (more so than neural networks), decision-tree based models, or learners. The algorithm builds on one learner at a time to fit the residual of the learner that preceded it, whose output is then summed to get an overall approximation. The trees learn when the approximation `ŷ` is moving closer to the target `y` by way of the minimization of the loss function `L`. We can think of the minimization, `y - ŷ`, as the residual difference between the target and approximation. This residual (the distance to the target `ŷ`) is a directional vector that points in the direction of the best approximation from the descent of the loss function.
 
So we can think of GBM's as a sum of two parts. It's one part a boosting algorithm that increases its output space---its second part is the boosting algorithm wrapped over a decision tree based architecture. The gradient descent mechanism of the algorithms optimization strategy happens on the output of the strong learners and not the parameters of the weak learners. The tree based architecture of the algorithm allows it to predict multiple, overlapping regions of the feature space. So they're good at finding strong, residual correlations between features.
 
I'll define some of the main ideas behind boosting using our binary classification problem where a scalar scoring function is formed to differentiate `bt_1` and `bt_5`. Given the data `X` denoted by <code>x<sub>i</sub></code>, and its labels `Y` denoted by <code>y<sub>i</sub></code>, the goal is to choose a classification function `F(x)` to minimize the aggregation of some specified loss function `L` or L(<code>y<sub>i</sub></code>, F(<code>x<sub>i</sub></code>))[36]. It's additive form would be:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104082095-386c5200-5201-11eb-95e1-33eaef3a5e41.png" width="420px">
</p>

<br/>

`argmin` or *"argument of the minimum"* permits the inputs minimum output, so it returns the value `F` which minimizes `L`. So `F` is our interest, and the objective is to find `L` so that the sum of its distance is as small as it can be, dependent on <code>x<sub>i</sub></code>.

Let's examine the function estimation `F` in gradient boosting:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104081865-96983580-51ff-11eb-8ec0-69ae45578791.png" width="280px">
</p>

<br/>

`T` is the number of iterations. <code>f<sub>m</sub>(x)</code> is designed cumulatively so that at the <code>m<sup>th</sup></code> stage, the recently calculated function, <code>f<sub>m</sub></code> will improve the total loss while retaining <code>{f<sub>j</sub>}<sup>m-1</sup><sub>j=1</sub></code> as a fixed property.      
 
Each function <code>f<sub>m</sub></code> is retained to a set of parameterized *"weak learners"*, letting `Θ` denote the vector of parameters of the weak learners. Gradient boosting uses decision trees as its weak learners, and because this `Θ` is composed of parameters that represent the tree structure that will split the feature in each internal node and also serves as the threshold for splitting each node, at phase `m`, we shape an estimated function of the loss:
 
<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104082207-5090a100-5202-11eb-834b-83c647fc33d5.png" width="550px">
</p>

<br/> 
 
<code>f<sub>m</sub></code> minimizes the right hand side of the second equation, and since the direction is only fitted to a shrinkage parameter which is designed to find the best step size, this is usually applied to tm before its added to <code>f<sub>m</sub>-1</code>. Where the derivative `∂` of the loss function `L` and the classification function `F`, defines the direction of `L`, its slope and optimizes how big of a step to take when finding the minimum. Hence its relation to gradient descent but is otherwise known as steepest descent. The advantage is that only the expression of the gradient varies for different loss functions, while the induction step of our weak learners remain the same for different loss functions. So essentially we can optimize the algorithm using any number of loss functions which also makes the algorithm diverse enough for classification or regression tasks[31].
 
One of the key ideas behind boosting algorithms is controlling bias and variance, which is necessary because tree based models induce high variance naturally. Bias is the sum of error from incorrect assumptions in the weak learners, and a large bias can cause the algorithm to overlook the pertinent connections between features and outputs. The variance is the sum of error of sensitivity from small changes in the same data. So a large variance will induce overfitting instead of finding the anticipated outputs.
 
Naturally, weak learners generate high bias and low variance... 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104083323-9ef66d80-520b-11eb-940c-fa0e3bc3c789.PNG" width="550px">
</p>

<br/> 

...and boosting reduces the output error by reducing bias and to a lesser extent variance, by aggregating the output from many sequential, weak learners so that their sum becomes one strong learner that reduces bias at every iteration. So the main advantage of xgb (XGBoost), is that we can use multiple models (weak learners) that will reduce variance and bias by training the model on the errors the previous learners made. 
 
But this method can also be optimized by introducing a random element to the algorithm's training, which has been proved to increase the accuracy of weak learners considerably, and they're computationally cost efficient which makes them easy to train, unlike neural networks. The randomization occurs when the weak learners are growing, and searching for the best split. Instead of making the split at the most distinct learning threshold, the `ExtraTreesClassifier` selects its split points fully at random for each feature, independent of the target variable and the best randomly generated threshold is chosen as the algorithm's splitting rule. Then they make a majority vote on the output based on the sum of weak learners. 
 
Randomization increases bias and variance of trees individually, but decreases their variance exponentially with respect to averaging over a large ensemble of trees, which naturally reduces high bias in the learning samples. Based on the accuracy in comparison to the neural network, this classification task can tolerate high levels of bias of class probability estimates without yielding high error rates. 

<br/>

# naive bayes

You can imagine the function of a support vector machine or even logistic regression drawing a line between or through some blob of data to separate the classes and classify them.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104111500-678dcc80-52b0-11eb-8c4e-0bd60ef18d73.png" width="410px">
</p>

<br/> 

With the probabilistic generative model naive bayes, you can imagine a probability distribution being drawn around the bits of each separate class so you end up with two separate probability distributions and in some cases a higher level of accuracy. 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104110025-a5cfbf80-52a1-11eb-9a82-5ee8cc0018f6.png" width="400px">
</p>

<br/> 

We need the data for `bt_1` and `bt_5` to learn a rule that maps an input `X` to an output `Y` given a number of training examples. In other words, we need to know what is the distribution of the `X`'s given the `Y`'s. The cool thing about the NB algorithm is that its function is baked into the name.

The problematic, naive part of NB double counts words because its independence assumption throws away the dependence that words have on each other. For example, the word `['Hong Kong']`, would be split into two separate words through tokenization, `['Hong'`,`'Kong']`, and is double counted, which can skew the final output. The independence assumption states that no features or words depend on each other, but this is counter intuitive to text as every word depends on prior words in very strict, grammatical hierarchies. 

The straightforward, bayesian part of NB lets us ask the computer a question in a way that we actually want the computer to represent it. We ask the question in the opposite way you want the answer returned, and we use Bayes Rule to flip the question the way that we want it. Before I begin to explain the algorithm used in the analysis, Bernoulli NB, let's consider a generic NB example. 

Given all the words in a bag, what is the probability that the document of tokens belongs to `bt_1`? The way you want to phrase it to the computer is given that the document of tokens belongs to `bt_1`, what is the probability that a given word will belong to `bt_1`. So you ask the question in reverse and use Bayes Rule to flip it back to a representation that we can understand.

NB is asking "What is the probability that an observation from the dataset is `bt_1` given the specific features `(X)` that we specify?". Let's break it down even further...

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104110236-acf7cd00-52a3-11eb-8294-f7fafc751d50.png" width="600px">
</p>

<br/> 

First we must calculate the prior probability `P(bt_1)`, which tells us the probability that a given observation is `bt_1` without knowing the amount of features or which features are associated to any of the other observations[32]. The only thing we can do is calculate all of `bt_1`'s observations and divide by the overall number of observations in the dataset.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104110337-c5b4b280-52a4-11eb-99f3-b4562ca72bc0.png" width="500px">
</p>

<br/> 

After calculating the prior probability, we'll need to calculate the marginal likelihood, `P(X)`. We need to take an input parameter of our choosing to build a radius around the unknown observation...

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104110075-2e4e6000-52a2-11eb-870d-65d74eab96ec.png" width="500px">
</p>

<br/> 

Looking at all the observations inside the radius, we're going to conclude that known observations are similar to unknown observation in terms of their position in the feature space. The probability of `X` is the probability of an unknown observation having similar features to the known observations within the parameter radius that we specify. `P(X)` is calculated as:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104110371-0ca2a800-52a5-11eb-9db1-487779f5fe89.png" width="550px">
</p>

<br/> 

Referencing the radius of data points from the graph above, we must compute the likelihood, `P(X|bt_1)`, which tells us the likelihood of a given observation belonging to `bt_1` given that they have the set of features specified by `X`. So, what is the likelihood that an unknown observation will be from the radius of observations with similar features to the unknown observation given (this is what the `|` stands for) that the observation is actually `bt_1`, which is calculated as:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104110620-67d59a00-52a7-11eb-838b-9fb3ad69d84f.png" width="550px">
</p>

<br/> 

When we plug these numbers into the equation we'll get the posterior probability, and it represents the final likelihood of a set of `bt_1`'s observations given `bt_1`'s features.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104110639-a9fedb80-52a7-11eb-905d-4582bc35e324.png" width="550px">
</p>

<br/> 

So we're left with a `67%` (which is better than random) probability that the unknown observation inside of our parameter radius `X` should be classified as one of `bt_1`'s observations. The algorithm will then do the same for `bt_5`'s observations.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104110676-006c1a00-52a8-11eb-8e9d-eb5149e129c7.png" width="550px">
</p>

<br/> 

Now we can compare the probability that the obsevation is `bt_1` given our features `X` vs. the probability that it could be `bt_5`. It's pretty obvious that `0.67 > 0.33`, so the question mark in the graph above will be `bt_1`'s observation rather than `bt_5`'s.

There are three versions of NB, but the one we're most interested in is Bernoulli NB, which focuses on the binary nature of the data. It tries to distinguish between the presence/absence of counts for a single class that occur and counts for the same class that do not occur. Since our target variable y can only belong to one of two classes, `bt_1` or `bt_5`, this means our target exclusively lies in the interval `0` or `1` and is represented by a binary feature vector.

So the sum of `bt_1` or `bt_5`'s documents are considered to be events and the presence and absence of words are considered attributes of the event. To compute the conditional probabilities for Bernoulli NB we will represent is as <code>P(bt_1|X)=P(<t<sub>1</sub>,...t<sub>k</sub>,...t<sub>n</sub>>|X)</code>, where <code><t<sub>1</sub>,...t<sub>k</sub>,...t<sub>n</sub>></code> is a paired vector of dimensionality `N` that indicates whether each term occurs in `bt_1` or not.
 
Remember the naive part of NB? The conditional independence assumption applies to Bernoulli NB because it still assumes that features are conditionally independent from one another, despite this assumption rarely being true, because each input you put into the model is assumed to not be dependent on the other inputs. In the case of Bernoulli NB, the conditional  independence assumption states:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104651826-32a1c100-5686-11eb-9c5f-4ab9cd4d20d1.png" width="900px">
</p>

<br/> 
 
To the right of the product, `P` representing the probability that a document of class `X` will occur if <code>e<sub>i</sub>=1</code> and will not occur if  <code>e<sub>i</sub>=0</code>. 

The most defining quality about Bernoulli NB is that it only takes into account the presence or absence of a word and does not capture the frequency of these words. While this may seem like there is a loss of ability as opposed to using a Multinomial NB or Gaussian NB, notice in the 46th notebook, our version of Bernoulli NB is also trained on words that are weighted by the TF-IDF -- GloVe embeddings. 
 
The semantic vectors preserve a fair amount of information about the text with relatively low dimensionality, which gives the algorithm an added advantage. Besides the features that were specified in `dataset['pos_features']`, the weighted semantic vectors allow Bernoulli NB to also capture semantic similarities of the text in `xtrain` when classifying a message to `bt_1` or `bt_5`.

<br/>

# 10. topic modeling
 
Earlier in the post we covered what distributions are, but before we can understand what topic modeling is and how its done, we should first make ourselves familiar, as qualitatively possible, with the dirichlet distribution. In order to to illustrate the unique nature of the dirichlet, lets consider the properties of a continuous normal distribution. Belonging to the same family of continuous probability distributions whose members include the t, logistic, and Laplace distributions, the ND... 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104653191-3c2c2880-5688-11eb-8237-80dcbeca83a3.gif" width="520px">
</p>

<br/> 


...represents the probability (*measure of spread*) over a set of real numbers supported by a whole line of integers and is defined by its mean (*target value*), variance (*the distributions width*), standard deviation (*the square root of the variance and tells us by how much the samples are expected to deviate from the mean*) and skewness of 0. If the standard deviation is high, you'll see values much larger or smaller than the mean, going in a very positive or negative direction. If the standard deviation is low, the samples will be very close to the mean.

The dirichlet distribution does not sample from a set of real numbers, it samples over a probability distribution that has conceptual ties to algebraic topology and it samples over something called a simplex. In all of our discrete number of possible topics in a given users text, we have a finite set of possibilities whose joint probabilities must sum to 1, with each probability represented as a point in Euclidean space. The probabilities will form a high dimensional triangle that is known as the simplex. When you observe the simplex...
 
<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104655495-c924b100-568b-11eb-8a5a-7937357ac369.gif" width="520px">
</p>

<br/>  

...imagine that the surface of the distribution is a continuous mixture of words in a document in Euclidean space, being mixed over a discrete number of possible topics inside the entire space. We can assume there are fundamental latent topics in the text from the `blues_traveler_2000` dataset, and each topic represents a multinomial distribution over the absolute value of words in the vocabulary at any point over the simplex.

We can segment varying combinations of topics and sample words from the continuous mix which will give us every probable multinomial distribution over all of the words, but the dirichlet will do so in such a way that it assigns each word a joint probability, made up of the mixture of topics and words, with values between `0` and `1` and the centroid of the simplex will be a uniform distribution over all words[34]. 

The heat-mapped surface of the simplex in the gif above represents the resulting density over the the sum of the multinomial distributions given by the generative model LDA (latent dirichlet allocation), that will use the dirichlet distribution to discover topics in our text. 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104656674-ad220f00-568d-11eb-80ea-dde89626aaa9.png" width="520px">
</p>

<br/>  

The vanilla LDA model in plate notation. The outer plate represents documents in the corpus, the inner plate represents the various topics in the document. `α` = topic distribution of document, `β` = word distribution of each topic, `θ` = topic distribution for plate `m`, `z` = topic for n<sup>th</sup> word in document `m`, `w` = a specific word.

Traditionally, topic modeling is a technique that's used to summarize legal documents, data mine customer support emails, match online ads to relevant webpages or to reduce the dimensionality of some `k` topic space for preprocessing tasks. But the application of topic modeling does not end with the analysis of a product, it can also be used to better understand text that someone has written in a non-formal setting. 

Like a diary for instance. On the flip side, based on the observations made in sections 5, 6 and 7, the immediate trade-off of predictions relating to words and their corresponding topics in this dataset leads me to assume that the Euclidean space will be very messy, because of the degrees of freedom associated with the stream ofconscious use of language, compared to the aforementioned industrial contexts which represent a more structured and finite topic space. 
 
So we must reduce the dimensionality of the text in such a way that LDA can capture something meaningful. Each users final text was preprocessed using 4 stopword lists. Three of the lists were canonical `(the, and, in, be etc.)` and the fourth was a customized, domain-specific `(bcuz, yep, yup, okkayyy, plllllleeeeaase, com, yeah etc.)` stopword list  and the combined lists totaled over 500 words. The text was lower cased, numerals were removed, the text was lemmatized, varying levels of n-grams ranging from no-grams to quin and sexgrams were used to capture phrases of words that were used together, and words that occurred in less than 20% of documents were also removed from each users corpora.
 
Finally, each users corpus was transformed into a bag-of-words and then used as input to a vanilla and labeled LDA model, the latter being a semi-supervised variant of the vanilla unsupervised LDA model. When using V-LDA, the interpretability of certain topics were affected by outlier words that were not associated to the majority of words in a given topic. 

To solve this, L-LDA was given a small set of priors in the form of words (seeds) as an indicator of what word would likely belong to a specific topic, which helped keep words without associations away from each other. This was made possible by a tunable parameter `Λ` that guided the model to concentrate around the defined priors or θ. While iteratively running the model, it became clear that some words were converging to topics in which they clearly did not belong; `love trust, spoon, marry, partner, especially`. This observation heavily influenced my judgement in defining L-LDA's seeds, but it's important to keep in mind that defining the seeds takes careful consideration and can reflect inherent biases. 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/104657345-cbd4d580-568e-11eb-8df4-ec85dd91734d.png" width="520px">
</p>

<br/>  

L-LDA model in plate notation. The representation of the plates is the same, but `α` = topic distribution of the document, `β` = word distribution for topic `k`, `n` = word distribution for each topic, `θ` = topic distribution for plate `m`, `z` = topic for <code>n<sup>th</sup></code> word in document `m`, `w` = a specific word. The label prior `Λ` and the seeds `Φ` guide are our new parameters and they guide the topic mixture in the direction of the seeds and give L-LDA a great upgrade for our use case.
