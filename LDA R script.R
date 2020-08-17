library(tm)
library(nlp)
library(stringr)
library(topicmodels)
library(tidytext)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(SnowballC)
library(textstem)

#################################################################################################################################
#IMPORTING DATASET

#I considered importing ".txt" files. Other types of files should be imported differently.

#Select folder with the documents
folder <- "G:\\Meu Drive\\...\\"

#select only ".txt" documents from the selected folder.
filelist <- list.files(folder, pattern = ".txt") 

#Join documents in an unique file
filelist <- paste(folder, "\\", filelist, sep="")   

#Consider each line as a different element (document).
x <- lapply(filelist, FUN = readLines)  

docs <- lapply(x, FUN = paste, collapse = " ")

#################################################################################################################################
#DATA CLEANING

#Remove punctuation
text <- gsub(pattern = "\\W", replace = " ", docs)
#Remove Numbers (digits)
text2 <- gsub(pattern = "\\d", replace = " ", text)
#Lowercase words
## Note: This is an essential procedure, since the text mining models cannot differentiate the same word with uppercase and lowercase forms. 
###E.g.: "Transport" and "transport" are considered two different words in the algorithm.  
text3 <- tolower(text2)
#remove single letter words 
text4 <- gsub(pattern = "\\b[A-z]\\b{1}", replace = " ", text3) 
#Remove whitespace
text5 <- stripWhitespace(text4)
#Lematize terms in its dictionary form
text6 <- lemmatize_strings(text5, dictionary = lexicon::hash_lemmas)
# Add adicional stopwords that were found by running the initial models. An example of terms that do not add value in academic journal papers is given.
adicional_stopwords <- c("table", "figure","study", stopwords("en"))
#remove stopwords
text7 <- removeWords(text6, adicional_stopwords)

#A Good practice is to visualize the corpus every now and then.
writeLines(as.character(text7[[1]]))

# Remove words for bigrams
new_stopwords <- c("ow", "ï", "exible", "cantly","wick", "â", "exibility", "uence", "uences", "ned")
text_bigram <- removeWords(text7, new_stopwords)

# Create corpus from vector
corpus <- Corpus(VectorSource(text7))

#################################################################################################################################
#TOPIC MODELLING - LATENT DIRICHLET ALLOCATION (LDA)

# Create document term matrix
dtm <- DocumentTermMatrix(corpus) 

#str(dtm)
#count top ten words
dtm.matrix <- as.matrix(dtm)
wordcount <- colSums(dtm.matrix)
topten <- head(sort(wordcount, decreasing=TRUE), 10)

#Number of topics (k) is defined prior to the model.
k <- 6

#Run LDA using Gibbs sampling
ldaOut <- LDA(dtm,
             k, 
             method="Gibbs", 
             control=list(seed = 42)) 
          
lda_topics <- ldaOut %>%
  tidy(matrix = "beta") %>%
          arrange(desc(beta))

lda_topics <- LDA(corpus,
              k, 
              method="Gibbs", 
              control=list(seed = 42))

#glimpse(ldaOut)

# select 15 most frequent terms in each topic
word_probs <- lda_topics %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  #Create term2, a factor ordered by word probability
  mutate(term2 = fct_reorder(term, beta))
  
# Plot term2 and the word probabilities
ggplot(
  word_probs,
  aes(term2,beta,fill = as.factor(topic))
) + geom_col(show.legend = FALSE) +
  # Facet the bar plot by topic
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  labs(x = "term")

#################################################################################################################################
#BIAGRAMS

library(quanteda)
library(igraph)
library(ggraph)
library(tidyr)

#Covert each paper to a line
t_corpus <- text_bigram %>% tidy()
#t_corpus

#create dataframe
df_corpus <- data.frame(t_corpus) 

#Create biagrams by separating words in sequences of 2. 
bigrams_df <- df_corpus %>%
  unnest_tokens(output = bigram,
                input = x,
                token = "ngrams",
                n = 2)

#Count bigrams
bigrams_df %>%
  count(bigram, sort = TRUE)

#Remove stopwords in case it wasn't in the beginning.
data("stop_words") 

#Separate words into two columns
bigrams_separated <- bigrams_df %>%
  separate(bigram, c("word1", "word2"), sep = " ")

#Remove stopwords
bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)

#Bigram counts, counts the number of times to words are always together
bigram_counts <- bigrams_filtered %>%
  count(word1, word2, sort = TRUE)

#Create network of bigrams

#filter for common combinations of biwords that appear at least 15 times.
bigram_network <- bigram_counts %>%
  filter(n > 15) %>%
  graph_from_data_frame()

set.seed(2016)

a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

ggraph(bigram_network, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = a, end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 4) +
  geom_node_text(aes(label = name), vjust = .7, hjust = 0.1) +
  theme_void()
