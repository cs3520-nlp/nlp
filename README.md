# Topic Modeling for Sentiment Analysis

This project aims to be able to use the concept of topic modeling to make an expert system that can predict a population's sentiment towards different movie descriptions. Our model uses [50k IMDB movie reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/activity) as its corpus.

* Note: **old_src** shows our initial work on this project, but it is not a part of the final program.

# How to Run:
 
In order to run our code, clone our repository and run **LDA_SA_Movie_Reviews_FINAL.ipynb** using Jupyter Notebook.
 
When asked if you'd like to use pickled files, enter "y" if you would like to use our pretrained models or "n" if you'd
prefer to create your own. If you choose to generate your own models, the appropriate model in the **pickled** folder will be overwritten with the new model. WARNING: generating a fresh model might take a long time.
 
When asked how many topics you wish to generate, please only enter "15", "35", "75", "150", or "300" as those are the only amounts that we've pickled.
