### Gather Data from organization/website ###
### Convert text data into DTM ###


########################################
## Topics: 
    # Data gathering via API
    #  - URLs and GET
    # Cleaning and preparing text DATA
    # DTM and Data Frames
    # Training and Testing at DT
    # CLustering
    ## LDA
    
#########################################    
    
    
## ATTENTION READER...
##
## First, you will need to go to 
## https://newsapi.org/
## https://newsapi.org/register
## and get an API key



################## DO NOT USE MY KEY!!
## Get your own key. 
##
#####################################


### API KEY  - get a key!
##https://newsapi.org/

## Example URL
## https://newsapi.org/v2/everything?
## q=tesla&from=2021-05-20&sortBy=publishedAt&
## apiKey=YOUR KEY HERE


import requests  ## for getting data from a server GET
import re   ## for regular expressions
import pandas as pd    ## for dataframes and related

## To tokenize and vectorize text type data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
## For word clouds
## conda install -c conda-forge wordcloud
## May also have to run conda update --all on cmd
#import PIL
#import Pillow
#import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
#from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz

from sklearn.decomposition import LatentDirichletAllocation 

from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn
from sklearn.cluster import KMeans

from sklearn import preprocessing

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram


####################################
##
##  Step 1: Connect to the server
##          Send a query
##          Collect and clean the 
##          results
####################################

####################################################
##In the following loop, we will query the newsapi servers
##for all the topic names in the list
## We will then build a large csv file 
## where each article is a row
##
## From there, we will convert this data
## into a labeled dataframe
## so we can train and then test our DT
## model
####################################################

####################################################
## Build the URL and GET the results
## NOTE: At the bottom of this code
## commented out, you will find a second
## method for doing the following. This is FYI.
####################################################

## This is the endpoint - the server and 
## location on the server where your data 
## will be retrieved from

## TEST FIRST!
## We are about to build this URL:
## https://newsapi.org/v2/everything?apiKey=1f5461bb65ca4c07afec9a19411d1ea9&q=bitcoin



topics=["politics", "football", "science"]



## topics needs to be a list of strings (words)
## Next, let's build the csv file
## first and add the column names
## Create a new csv file to save the headlines
filename= r"C:\Users\enki0\Desktop\CSCI5922\Final_Exam\NewHeadlines1.csv"
MyFILE=open(filename,"w")  # "a"  for append   "r" for read
## with open
### Place the column names in - write to the first row
WriteThis="LABEL,Date,Source,Title,Content\n"
MyFILE.write(WriteThis)
MyFILE.close()

## CHeck it! Can you find this file?
    
#### --------------------> GATHER - CLEAN - CREATE FILE    

## RE: documentation and options
## https://newsapi.org/docs/endpoints/everything

endpoint="https://newsapi.org/v2/everything"

################# enter for loop to collect
################# data on three topics
#######################################

for topic in topics:

    ## Dictionary Structure
    URLPost = {'apiKey':'1f5461bb65ca4c07afec9a19411d1ea9',
               'q':topic,
              # 'source': 'bbc-news',
              # 'pageSize': 100,
              # 'sortBy' : 'top',
              # 'totalRequests': 1000,
               #'from':'2023-12-05',
              # 'to':'2000-01-01' 
    }

    response=requests.get(endpoint, URLPost)
    print(response.content)
    jsontxt = response.json()
    print(jsontxt)
    #####################################################
    
    
    ## Open the file for append
    MyFILE=open(filename, "a")
    LABEL=topic
    for items in jsontxt["articles"]:
        print(items, "\n\n\n")
                  
        #Author=items["author"]
        #Author=str(Author)
        #Author=Author.replace(',', '')
        
        Source=items["source"]["name"]
        print(Source)
        
        Date=items["publishedAt"]
        ##clean up the date
        NewDate=Date.split("T")
        Date=NewDate[0]
        print(Date)
        
        ## CLEAN the Title
        ##----------------------------------------------------------
        ##Replace punctuation with space
        # Accept one or more copies of punctuation         
        # plus zero or more copies of a space
        # and replace it with a single space
        Title=items["title"]
        Title=str(Title)
        #print(Title)
        Title=re.sub(r'[,.;@#?!&$\-\']+', ' ', str(Title), flags=re.IGNORECASE)
        Title=re.sub(' +', ' ', str(Title), flags=re.IGNORECASE)
        Title=re.sub(r'\"', ' ', str(Title), flags=re.IGNORECASE)
        
        # and replace it with a single space
        ## NOTE: Using the "^" on the inside of the [] means
        ## we want to look for any chars NOT a-z or A-Z and replace
        ## them with blank. This removes chars that should not be there.
        Title=re.sub(r'[^a-zA-Z]', " ", str(Title), flags=re.VERBOSE)
        Title=Title.replace(',', '')
        Title=' '.join(Title.split())
        Title=re.sub("\n|\r", "", Title)
        print(Title)
        ##----------------------------------------------------------
        
        Content=items["description"]
        Content=str(Content)
        Content=re.sub(r'[,.;@#?!&$\-\']+', ' ', Content, flags=re.IGNORECASE)
        Content=re.sub(' +', ' ', Content, flags=re.IGNORECASE)
        Content=re.sub(r'\"', ' ', Content, flags=re.IGNORECASE)
        Content=re.sub(r'[^a-zA-Z]', " ", Content, flags=re.VERBOSE)
        ## Be sure there are no commas in the headlines or it will
        ## write poorly to a csv file....
        Content=Content.replace(',', '')
        Content=' '.join(Content.split())
        Content=re.sub("\n|\r", "", Content)
        
        ### AS AN OPTION - remove words of a given length............
       # Content = ' '.join([wd for wd in Content.split() if len(wd)>1])
    
        #print("Author: ", Author, "\n")
        #print("Title: ", Title, "\n")
        #print("Headline News Item: ", Headline, "\n\n")
        
        #print(Author)
        print(Title)
        print(Content)
        
        WriteThis=str(LABEL)+","+str(Date)+","+str(Source)+","+ str(Title) + "," + str(Content) + "\n"
        print(WriteThis)
        
        MyFILE.write(WriteThis)
        
    ## CLOSE THE FILE
    MyFILE.close()
    
################## END for loop

####################################################
##
## Where are we now?
## 
## So far, we have created a csv file
## with labeled data. Each row is a news article
##
## - BUT - 
## We are not done. We need to choose which
## parts of this data to use to model our decision tree
## and we need to convert the data into a data frame.
##
########################################################


BBC_DF=pd.read_csv(filename, on_bad_lines='warn',encoding='unicode_escape')

print(BBC_DF.head())
# iterating the columns 
for col in BBC_DF.columns: 
    print(col) 
    
print(BBC_DF["Content"])

## REMOVE any rows with NaN in them
BBC_DF = BBC_DF.dropna()
print(BBC_DF["Content"])

### Tokenize and Vectorize the Headlines
## Create the list of headlines
## Keep the labels!

ContentLIST=[]
LabelLIST=[]

for nexttext, nextlabel in zip(BBC_DF["Content"], BBC_DF["LABEL"]):
    ContentLIST.append(nexttext)
    LabelLIST.append(nextlabel)

print("The headline list is:\n")
print(ContentLIST)

print("The label list is:\n")
print(LabelLIST)


##########################################
## Remove all words that match the topics.
## For example, if the topics are food and covid
## remove these exact words.
##
## We will need to do this by hand. 
NewContentLIST=[]

for element in ContentLIST:
    print(element)
    print(type(element))
    ## make into list
    AllWords=element.split(" ")
    print(AllWords)
    
    ## Now remove words that are in your topics
    NewWordsList=[]
    for word in AllWords:
        print(word)
        word=word.lower()
        if word in topics:
            print(word)
        else:
            NewWordsList.append(word)
            
    ##turn back to string
    NewWords=" ".join(NewWordsList)
    ## Place into NewHeadlineLIST
    NewContentLIST.append(NewWords)


##
## Set the     HeadlineLIST to the new one
ContentLIST=NewContentLIST
print(ContentLIST)     
#########################################
##
##  Build the labeled dataframe
##
#########################################

### Vectorize
## Instantiate your CV
MyCountV=CountVectorizer(
        input="content",  ## because we have a csv file
        lowercase=True, 
      #  stop_words = "english",
        max_features=1000
        )

## Use your CV 
MyDTM = MyCountV.fit_transform(ContentLIST)  # create a sparse matrix
print(type(MyDTM))

# This is vocab
ColumnNames=MyCountV.get_feature_names_out()
#print(type(ColumnNames))

## Build the data frame
MyDTM_DF=pd.DataFrame(MyDTM.toarray(),columns=ColumnNames)

## Convert the labels from list to df
Labels_DF = pd.DataFrame(LabelLIST,columns=['LABEL'])

## Check your new DF and you new Labels df:
print("Labels\n")
print(Labels_DF)
print("News df\n")
print(MyDTM_DF.iloc[:,0:3])

##Save original DF - without the lables
My_Orig_DF=MyDTM_DF
print(My_Orig_DF)
######################
## AND - just to make sure our dataframe is fair
## let's remove columns called:
## food, bitcoin, and sports (as these are label names)
######################
#MyDTM_DF=MyDTM_DF.drop(topics, axis=1)


## Now - let's create a complete and labeled
## dataframe:
dfs = [Labels_DF, MyDTM_DF]
print(dfs)

Final_News_DF_Labeled = pd.concat(dfs,axis=1, join='inner')
## DF with labels
print(Final_News_DF_Labeled)


#############################################
##
## Create Training and Testing Data
##
#############################################


## Before we start our modeling, let's visualize and
## explore.

##It might be very interesting to see the word clouds 
## for each  of the topics. 
##---------------------------------------------------
List_of_WC=[]

for mytopic in topics:

    tempdf = Final_News_DF_Labeled[Final_News_DF_Labeled['LABEL'] == mytopic]
    print(tempdf)
    
    tempdf =tempdf.sum(axis=0,numeric_only=True)
    #print(tempdf)
    
    #Make var name
    NextVarName=str("wc"+str(mytopic))
    #print( NextVarName)
    
    ##In the same folder as this code, I have three images
    ## They are called: food.jpg, bitcoin.jpg, and sports.jpg
    #next_image=str(str(mytopic) + ".jpg")
    #print(next_image)
    
    ## https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
    
    ###########
    ## Create and store in a list the wordcloud OBJECTS
    #########
    NextVarName = WordCloud(width=1000, height=600, background_color="white",
                   min_word_length=4, #mask=next_image,
                   max_words=200).generate_from_frequencies(tempdf)
    
    ## Here, this list holds all three wordclouds I am building
    List_of_WC.append(NextVarName)
    

##------------------------------------------------------------------
print(List_of_WC)
##########
########## Create the wordclouds
##########
fig=plt.figure(figsize=(25, 25))
#figure, axes = plt.subplots(nrows=2, ncols=2)
NumTopics=len(topics)
for i in range(NumTopics):
    print(i)
    ax = fig.add_subplot(NumTopics,1,i+1)
    plt.imshow(List_of_WC[i], interpolation='bilinear')
    plt.axis("off")
    plt.savefig(r"C:\Users\enki0\Desktop\CSCI5922\Final_Exam\NewClouds.png")
 
     
###############################################################         
## STEP 1   Create Training and Testing Data
###############################################################
## Write the dataframe to csv so you can use it later if you wish
##
Final_News_DF_Labeled.to_csv(r"C:\Users\enki0\Desktop\CSCI5922\Final_Exam\Labeled_News_Data_from_API1.csv")
# TrainDF, TestDF = train_test_split(Final_News_DF_Labeled, test_size=0.3)
# print(TrainDF)
# print(TestDF)

#################################################
## STEP 2: Separate LABELS
#################################################
## IMPORTANT - YOU CANNOT LEAVE LABELS ON 
## Save labels

### TEST ---------------------
# TestLabels=TestDF["LABEL"]
# print(TestLabels)
# TestDF = TestDF.drop(["LABEL"], axis=1)
# print(TestDF)
# ### TRAIN----------------------
# TrainLabels=TrainDF["LABEL"]
# print(TrainLabels)
# ## remove labels
# TrainDF = TrainDF.drop(["LABEL"], axis=1)
