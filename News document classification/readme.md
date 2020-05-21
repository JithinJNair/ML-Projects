{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Business Problem"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Description"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Text documents are one of the richest sources of data for businesses: whether in the shape of customer support tickets, emails, technical documents, user reviews or news articles, they all contain valuable information that can be used to automate slow manual processes, better understand users, or find valuable insights. However, traditional algorithms struggle at processing these unstructured documents, and this is where machine learning comes to the rescue!\n",
    "\n",
    "We’ll use a public dataset from the BBC comprised of 2225 articles, each labeled under one of 5 categories."
   ]
  },
  {
   "attachments": {
    "Automatic%20Document%20classification%20machine%20learning.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Automatic%20Document%20classification%20machine%20learning.png](attachment:Automatic%20Document%20classification%20machine%20learning.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-05T09:20:32.893948Z",
     "start_time": "2020-05-05T09:20:32.890945Z"
    }
   },
   "source": [
    "###### Problem Statement"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- We are tasked with predicting the category of a news article using machine learning models.\n",
    "- This could be useful to instantly predict category of an article ,for which it is not mentioned.\n",
    "  Helpful in reducing manual task."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Sources or Useful Links"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Source1: http://mlg.ucd.ie/datasets/bbc.html - Train and Validation data set.\n",
    "- Source2: https://inshorts.com/ - Evaluation data set.(Collected through WebScrapping)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Real World/Business objectives and Constraints"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-05T09:36:32.558686Z",
     "start_time": "2020-05-05T09:36:32.553682Z"
    }
   },
   "source": [
    "- Cost of misclassification can be high.\n",
    "- Interpretability is partially important"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Machine Learning Problem"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Overview"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-05T09:44:41.929330Z",
     "start_time": "2020-05-05T09:44:41.923328Z"
    }
   },
   "source": [
    "- Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.\n",
    "- Class Labels: 5 (business, entertainment, politics, sport, tech).\n",
    "- Size of bbc file:4.80MB.\n",
    "- Scrapped data from https://inshorts.com/ for evaluation dataset.\n",
    "- inshorts dataset contains 25 articles of 5 (business, entertainment, politics, sport, tech)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Mapping Real World problem to Machine Learning problem"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Multiclass classification problem.For a given article we need to predict its caetgory among five classes.\n",
    "- Train the model on BBC data.The model should be able to categorize article from any news source."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Metrics"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2020-05-05T09:57:00.344894Z",
     "start_time": "2020-05-05T09:57:00.338890Z"
    }
   },
   "source": [
    "- Multiclass Confusion matrix- Macro avg F1score.\n",
    "- Macro avg F1score=2*Pr_macro⋅Re_macro/(Pr_macro+Re_macro)\n",
    "- If F1macro has a large value, this indicates that a classifier performs well for each individual class. \n",
    "  The macro-average is therefore more suitable for data with an imbalanced class distribution."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Pre processing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- In this session we performed\n",
    "    - Null value treatment.\n",
    "    - Html&URL removal\n",
    "    - Decontraction of words\n",
    "    - lower case conversion\n",
    "    - special characters removal\n",
    "    - stop words removal."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Feature Extraction steps"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Extracted first line of each articles and added a new featue-title.\n",
    "- Gave more weightage to words in title."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Vectorization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Performed Tfidf-vectorization with min_df=5 and ngram=(1,2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Select-K best"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Used Selct K best method to get best 6000 features ,based on chi2 value .\n",
    "- Helped to reduce dimentions and reduce impact of overfitting."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model Building ,validation and Evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Performed model validation using unseen bbc articles with Logistic regression,SVM and Random Forest methods.\n",
    "- Performed model evaluation on data scrapped from inshorts.com with Logistic regression,SVM and Random Forest methods.\n",
    "- SVM is selected for Document classiffier application development."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary & Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- We created train and validation data with articles from bbc.\n",
    "- Extracted title from articles and added as a new feature.\n",
    "- Treated the text features for htmls,urls,contracted words,stop words etc.\n",
    "- Tested the model performance on train and validation data.\n",
    "- Its been identified that the model is performing well on both train and validation data with accuraccy & F1_score of 100% on train \n",
    "  and 97%on test.\n",
    "- Upon further analysis,its been understood that the high performance is due to facts that,\n",
    "    - There are very few data points available.Which makes our model to overfit on data.This cause high accuraccy on train data.\n",
    "    - both the train and validation set article belongs to bbc.Hence,the language structure of the articles may be the same.Which cause train and test data to be in similar to nature.And hence,high accuraccy on test data too.\n",
    "- For forther evaluation,we webscrapped inshorts.com ,collected the articles from 5differrent categories and created an evaluation dataset.\n",
    "- On evaluation dataset,model is giving high F1_Score of 100% on train and 76% on evaluation data.Which suggestec the model is overfitting.\n",
    "- The model as not able to perform well with articles related to business and tech.\n",
    "- To tackle overfittingwe gave more weightage to tile.Because,title itself contains lot of information.\n",
    "- Selected best 6000 features using chi2 metric.\n",
    "- On selected 6000 features,the model performance improved,eventhough that is not a substantial improvement.\n",
    "- The model performace improved from 76% to 78% f1_score.\n",
    "- The models ability to identify tech articles improved.\n",
    "- The model performnce can be improved by increasing training records/samples.\n",
    "- Training the model with articles from differrent sources can increase its ability to generalize on unseen data.\n",
    "- Selecting best 6000 features can reduce the impact of overfitting.\n",
    "- We perfomed model validation and evaluation with logistic regression,SVM and Random forest and OneVsRest approach.\n",
    " All three algorithms provided similar results.\n",
    "- SVM with best 6000 features are selected for creating Document classifiaction application.\n",
    "- Tkinter is selected to develop the front end."
   ]
  }
 ],
 "metadata": {
  "hide_input": false,
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {
    "height": "calc(100% - 180px)",
    "left": "10px",
    "top": "150px",
    "width": "165px"
   },
   "toc_section_display": true,
   "toc_window_display": true
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
