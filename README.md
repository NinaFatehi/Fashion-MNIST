# Fashion-MNIST
In this repository, Singular Value Decomposition (SVD), K-means is applied on Fashion-MNIST dataset.

- SVD function is used to reduce the number of dimensions of the training data
set so that it explains just above 90% of the total variance.

- Generative classifiers (Naive Bayes and KNN) and discriminative classifier
(multinomial logistic regression) are trained on both the training data set after SVD and the
original data set (without dimension reduction). 
