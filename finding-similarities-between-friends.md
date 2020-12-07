# abstract

With the surge of social media, the internet has become an interesting and spirited domain in which billions of individuals from all around the world interact, share, post and conduct many daily activities. The immense size of social media data makes it notably different from classic data sources, and the mainly user generated data can be unbelievably noisy and unstructured. In social media mining, social media is considered a world of social atoms (i.e. *individuals*), and entities (e.g. *content, sites, networks etc.*) signaling interactions between social atoms and entities (et al. *Zafarani, Abbasi and Liu*). In this analysis I propose a way of looking at interactions governed solely between social atoms by collecting, mining and measuring the interactions between these social atoms to discover whatever salient patterns reside in each atom as they are projected through the lens of social media.

It is my goal to make sense of a Facebook Messenger chat group belonging to a group of friends that has been active for about 8 years and I hope to find relationships within each members text to derive an overall view of what those relationships mean to each other. The shared meaning within these conversations go back pretty far, and should be reinforced when you look over the entire body of conversational sentences. I don't use the word *meaning* as a way to represent strings of words relating to the intent of a speaker per say. The meaning I'm after in this analysis is derived *from* form rather than *context* of use. This is to say that we will learn which words are similar in place of commonsense reasoning without truly knowing what each atoms' private mental state infers. 

The overall text fragments within this corpus will be very short, considering the conversations held within the medium are stream of conscious by nature. I will explain in great detail, methods of deriving meaning from these short series of sentences using dense, distributed word embeddings by way of deep learning. After the text has been cleaned, each word will be mapped onto points in a high dimensional space to further reduce *meaningless* words to obtain *meaningful* words for each user. During the text extraction about 25,251 instances from the sum of each users name--originally ending at the 302,733rd instance--was lost. In an attempt to label this wealth of data so that it can be included in modeling user sentiment and topics, I propose a binary classification task to classify each users name to their respective text using an attention-based bidirectional long, short term memory recurrent neural network to learn the relevant features of each users text. The AB-BiLSTMRNN results will be weighted against engineered features tested on more classical information retrieval and shallow learning techniques such as term frequency inverse document frequency, sparse count vectorization, logistic regression, gradient boosted trees, naive bayes and support vector machines.

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

<img src="https://user-images.githubusercontent.com/29679899/101290745-2f3b3000-37d2-11eb-813f-8cf472ba58dd.PNG" width="500px">

<br/>

...that brings together mathematics, computer science and  linguistics to make human language attainable for computers. Natural language processingshares ideas with computational linguistics, but CL is in service of linguistics... 

<br/>

<img src="https://user-images.githubusercontent.com/29679899/101290912-ff405c80-37d2-11eb-91e0-071835432ba9.PNG" width="500px">

<br/>

...while NLP is centered around the design and exploration of different representations for parsing natural language.  

In Katzian semantics--a form of generativist structural semantics--its said that word meanings are defined in terms of their combination of simpler conceptual components, therefore word meanings are structured entities whose semantic markers reproduce the structure of the represented meaning and whose labels are the words conceptual components. Katzian semantics has its flaws[<a href="https://plato.stanford.edu/entries/word-meaning/#GenSem" title="Katzian semantics" rel="nofollow">'</a>], but the idea that words and their meanings can belong to a structure that lends itself to analysis is still very relevant.  

In an attempt to illustrate the idea of meaning, let's put the words `car`, `bus`, `road` and `driving` into a bag, mix them all up and dump them on the floor. Looking at the words, what do we see? The meaning between a group of words is equal to the distance between them, based on the likeness of their meaning[1]. I'm not just talking about words that appear similar to each other as opposed to just similarity--which can be estimated based on a set of rules, principles and processes that govern the structure of sentences in a given language--what I'm saying is that no matter what order the words are in, `car` will be similar to `bus` and both words are related to `road` and `driving`. This specific type of similarity is the building block of various mathematical tools that are used to estimate the strength of the semantic relationship between each unit of language.

Let's say you want to look at the way a person writes. You could very well end up asking questions like, *"What words are they using? How often are they using similar words?"*. Term frequency-inverse document frequency can be described as a method that looks at a persons vocabulary. Term frequency is how often you use particular words and inverse document frequency is how rare those words are across the document as a whole. The idea is that as people use certain words that aren't common, those are the words that are particularly strong signatures in that individuals writing style. So we'rebasically able to analyze that individuals' vocabulary. 

This method is far from perfect and it ignores some of the most fundamental ideas of meaning. If you're just counting the frequency of words, we're missing out on why those words connect, which is pretty crude. The frequency of a word is somewhat misleading because you have to take into account how words appear together to form something meaningful. When we think of natural language processing as a way to potentially overcome this, we must also make computational linguistic goals. I've taken a sample sentence from one users corpus and used it as input to a linguistics parser built by the NLP Group at Stanford. This program is able to go through the sentence and establish a words syntactic role.

*"Every other perception of soccer in the US is sitting through a grueling game of 9 years olds running in circles."*

<br/>

<img src = "https://user-images.githubusercontent.com/29679899/101291038-f00dde80-37d3-11eb-8532-cdf158534cac.png" width="500px">

<br/>

This parse tree was trained by a language model, and its looking for various parts of speech within the sentence by looking over all noun and verb phrases. 

A core aspect of NLP from a machine learning/statistical perspective are language models. Language models help machines learn that certain words and sentences are more probable than other words and sentences. So a LM could infer that *My pots and pans are in the dishwasher* is more probable than *My pots and pans are in the school bus*. 

To simplify how the LM behind this dependency parser focuses on the relationships between the words in the *soccer* sentence, take a look at the part of speech (pos) tags for each word and you can see what role each word plays within the sentence...

<br/>

<img src="https://user-images.githubusercontent.com/29679899/101291138-94902080-37d4-11eb-82a6-6b13efc3bb51.PNG" width="500px">


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

<img src="https://user-images.githubusercontent.com/29679899/101291371-3feda500-37d6-11eb-900d-e9fc26c08faa.PNG" width="500px">

<br/>

After Facebook authorized the download, I was able to go through most of the data I've put on their platform over the years, but the only thing I'm interested in is the wealth of texts from Facebook Messenger that I've accumulated in this group:

<br/>

<img src="https://user-images.githubusercontent.com/29679899/101291409-82af7d00-37d6-11eb-95b3-d58f9fbde990.PNG" width="200px">

<br/>

All of the messages for my intended corpus are now accessible through a nice little (98.4MB) html file. As I scroll through this html file in the browser, I notice a few things. Each users response to the messenger app contains four pieces of data. Their names (which will remain blurred), message and the date/time in which they made a response on the app. Simple enough: 

<br/>

<img src="https://user-images.githubusercontent.com/29679899/101291432-aa064a00-37d6-11eb-9dc3-1283d94e85bc.PNG" width="500px">

<br/>

Web pages are nice when you're browsing the internet, but analyzing the text would be a lot easier if I could aggregate each persons message into a row that corresponds with their name, date and time of their response. Understanding the logic of the html structure is necessary before any information can be extracted, so if we view the html page's source we can get a closer look.

<br/>

![giphy](https://user-images.githubusercontent.com/29679899/101291471-02d5e280-37d7-11eb-934b-bb7e9c56d99b.gif)

<br/>

In the gif above all the information we need is nested within div tags. A div tag is simply a container that encloses page elements in an html file and divides the html file into sections. What we're interested in is what's inside the div containers. Specifically the user names, which are in the `"_3-96 _2pio _2lek _2lel"` div class, their messages are in the `"_3-96 _2let"` div class and the date & time is in the `"_3-94 _2lem"` div class. The div classes--which is where your distinctive markers will usually reside--within the nested div tags contain unique identifiers that will allow us to extract all the div containers that have class attribute `"_3-96 _2pio _2lek _2lel"`, `"_3-96 _2let"` or `"_3-94 _2lem"`. We can scrape each users text, dates and times from this web page, because each piece of information is nested within a particular variable. To scrape the text, we will use a common parsing module called BeautifulSoup.

```python

# dependencies
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
<img src="https://user-images.githubusercontent.com/29679899/101291640-509f1a80-37d8-11eb-927f-3f6207fd270b.PNG" width="450px">

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

# 2. initial data exploration 

Datasets are like a good satirical bildungsroman[<a href="https://en.wikipedia.org/wiki/Bildungsroman" title="What is a bildungsroman" rel="nofollow">'</a>]. They will give you many ambiguous ideas, and sometimes they will even have an interesting story to tell. Exploring your data is a very important pillar of data analysis, because it gives a sense of what can be done with it and what may be possible. The first few reduction methods in this post will attempt to deconstruct the signal in our dataset into a set of features that will help our algorithms learn something meaningful. The only way you can handcraft good features for your data is by visualizing it, and after extracting the text from the html file and exporting everything into a nice tabular format, our first visualization is just a matter of several dependencies and less than 10 lines of code.

```python

# dependencies
import pandas as pd
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
dataset = pd.read_csv('bt_fb_messenger_data.csv').fillna('')
# shape of data
print("Training Data Shape : ", dataset.shape)

```

`pd.read_csv` is a function in `pandas` that allows data to pass into python, but unlike python's native `.read()` function, we can perform containerization operations that are exclusive and specific to the pandas library. `pd.read_csv` is contained within the dataset variable which is then used to determine the shape of the data using the .shape function. After executing these two lines of code, a `pandas` matrix is returned. A matrix in pandas is synonymous with arrays in python. Arrays are data structures that can hold any rendition of data in a structured way, and a matrix is simply a two-dimensional data structure where numbers are arranged into rows and columns. The returned integer tells us that there are 302,731 rows, within the table and 4 columns in a 2-dimensional state. 

In order to understand the text, we'll first need to get a sense for how the variables are distributed. Distributions show us the possible values for variables and how often they occur. Maybe we want to compare the distribution of a variable across levels of other variables. Are we working with univariate, bivariate or multivariate distributions? You can easily make some of these assumptions by just looking at the data in its raw form, but visualizing it helps us to understand it on a more intuitive level. 

Since each users response in the app is dictated by the amount of times they send a message, lets obtain the frequency of each users response. We'll isolate the feature `Names` from the dataset in the names variable to determinehow many total occurrences there are for each user. To visualize the frequency, we'll specify the plots height and width with the figsize element and call it from the `plt.figure` function.

`sns.barplot` supports the input data from `.index`, which specifies where the index values begin, and we use this to access the `pandas` data frame within the names variable, and `.values` allows the index values to be displayed. alpha specifies a semi-translucent tone to the bar plot and color simply returns a color for the bars. `plt.xlabel` and plt.ylabel applies a label to the `x/y` axis and `plt.show()` allows us to visualize the 6 lines of code.
