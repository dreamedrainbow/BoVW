# image-classifier
Classify images using logistic regression on bags of visual words.  Currently,
the features extracted are:

- The standard Haralick texture features.
- Local features obtained via SURF that are then clustered (using k-means).  Each cluster represents a visual word.  An image is then realized as a bag of visual words by counting how many SURF features belong to each cluster.  This is similar to how one might approach topic assignment for textual documents.
