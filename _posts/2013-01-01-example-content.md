---
layout: post
title: "Iris Flower DataSet: Visualizing a Decision Tree"
date: "2016-10-13"
slug: "Iris_dataset_solution"
description: "A very basic beginner guide to visualize the one of the classic and beginner level data set available i.e 'The Iris Flower Dataset' to predict the label for new flower using Python Libraries like scikit-learn, NumPy etc."
category: 
  - views
  - featured
# tags will also be used as html meta keywords.
tags:
  - Iris Flower Data Set
  - solution
  - scikit-learn
  - Data Analysis
show_meta: true
comments: true
mathjax: true
gistembed: true
published: true
noindex: false
nofollow: false
# hide QR code, permalink block while printing.
hide_printmsg: false
# show post summary or full post in RSS feed.
summaryfeed: false
## for twitter summary card with squared image and page description or page excerpt:
# imagesummary: foo.png
## for twitter card with large image:
# imagefeature: "http://img.youtube.com/vi/VEIrQUXm_hY/0.jpg"
## for twitter video card: (active for this page)
videofeature: "https://www.youtube.com/embed/iG9CE55wbtY"
imagefeature: "http://img.youtube.com/vi/iG9CE55wbtY/0.jpg"
videocredit: tedtalks
---

# Iris Data Set
The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Ronald Fisher in his 1936 paper 'The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis'.

This data sets consists of 3 different types of irises' (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray

The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width. The details of the Data set can be found [Here](https://en.wikipedia.org/wiki/Iris_flower_data_set)

<!--more-->

## Goals for the Data Set

To analysize the Dataset we will follow the following set of procedure. Make sure you try and understand each of it.

- **Import Dataset:** The first step is to import the Iris data set into our code. Fortunately, scikit-learn comes with this preloaded Dataset, so we just need to import it. Load the dataset form the official scikit learn website.
[Scikit-learn docs](http://scikit-learn.org/stable/datasets/) 
- **Train a classifier** Using the dataset, we will train a classifier.
- **Predict label for new flower** We'll use the classifier to predict that what species of flower we have.
- **Visualize the tree** Visualization of the tree using DecisionTreeClassifier class and DecisionTreeRegressor class.


## Import code and Understanding the Dataset


{% highlight python linenos%}
// Load the dataset form scikit-learn
from sklearn.datasets import load_iris

iris = load_iris()

// Printing the different features/parameters of the Dataset
// Print features_names
print(iris.features_names)
print(iris.target_names)
print(iris.data[0])
print (iris.target[0])

{% endhighlight %}

Now we have printed the features of the flower on the basis of which we are going to classify the problem. The different features are 'Sepal length', 'Sepal width', 'Petal length' and 'Petal width'. And also the target variables contains the labels.

Now lets see the complete Data Set

{% highlight python linenos%}
for i in range (len(iris.target)):
  print ("Examples %d: label %s, features %s" % (i, iris.target, i], iris.data [i]))

{% endhighlight %}

The data set contains details of 150 Iris Flowers and we just printed them in a tabular form.


## Train a classifier and predict label for new flower:

We first need a testing data, A testing data are the examples used to 'Test' the classifier's accuracy, these are not the part of the actual training dataset. Sometimes, while working with huge datasets it's often the case that it takes a lot of time to actually train our data without the surety that our classifier algorithms works fine. So it's always a good practise to 'Test' the data before traning it. Testing, unlike in programming, is a very necassry and important element of Machine Learning.

{% highlight python linenos%}
import numpy as np
from sklearn import tree
test_idx = [0,50,100]

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx)

#Testing Data, for testing data we are only focused on test_idx
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#Creating decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)
print (clf.predict(test_data))

{% endhighlight %}

For details about NumPy array, refer to the [docs](https://docs.scipy.org/doc/numpy/reference/)

np.delete(name of array from where the data is to be deleted, the subarray that is to be deleted) Refer NumPy docs.
We deleted 3 entries from our training data for testing. We created a decision tree and fit the training data into it. And then we checked that our training data is similar to the testing data or not


## Visualize the tree.

{% highlight python linenos%}
from IPython.display import Image
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png()) 

// If we have pydotplus installed, we can generate a PDF file directly in Python:
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
{% endhighlight %}


## Decision trees using DecisionTreeRegressor class:

Decision trees can also be applied to regression problems, using the DecisionTreeRegressor class.

As in the classification setting, the fit method will take as argument arrays X and y, only that in this case y is expected to have floating point values instead of integer values:

{% highlight python linenos%}

from sklearn import tree
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])
array([ 0.5]
{% endhighlight %}



### References

Refer to the following docs/videos for more information on how to apply Machine Learning to Iris Data Set.

1: [scikit-learn Official docs](http://scikit-learn.org/stable/tutorial/index.html)

2: [NumPy Official docs](https://docs.scipy.org/doc/numpy/reference/)

3: [Google Developer Video on Machine Learning](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal)

Tip for beginner: Install jypyter notebook via [Anaconda](https://www.continuum.io/downloads), as it comes with almost all the preloaded libraries and stuff needed for learning basic Machine Learning concept.

Feel free to drop a comment below for any queries/comments.
Happy coding :)