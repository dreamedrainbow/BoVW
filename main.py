from __future__ import print_function
import numpy as np
import mahotas as mh
import itertools
from glob import glob
from mahotas.features import surf
from sklearn import cross_validation
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.externals import joblib # for load/dumping classifiers
from sklearn.cluster import KMeans
from sklearn import svm
#from mpltools import style
from matplotlib import pyplot as plt
from tagdownload import DATA_DIR 

#TODO 
# -Store the features for future re-use?

# -Given an image file and tags associated to said image,
# write a function that classifies the image and
# suggests that it be tagged with the class tag if
# that is not present.  If tags are present on
# the image that are included in the class list
# that don't correspond to high probabilities,
# suggest that they are removed.


def get_labels(classes):
    """
    Get the class labels for the training set.
    
    Keyword arguments:
    classes (list)- a list of strings consisting of the class names
    
    Returns:
    images (list)  - a list of strings consisting of the names of the images
    in the training data.  The training data is assumed
    to be located in the appropriate subdirectories of
    DATA_DIR.
    
    labels (numpy array of ints) - an numpy integer array consisting of the 
    numeric class labels for the training data.  
    """
    images = []
    labels = []
    
    for classIndex, className in enumerate(classes):
        current_images = glob('{}/{}/*.jpg'.format(DATA_DIR,className))
        images.extend(current_images)
        labels.extend([classIndex for image in current_images])
    
    labels = np.array(labels)
    
    return images, labels

def get_standard_features(images):
    """
    Return the Haralick features for each image in images.
    
    Keyword arguments:
    images - a list of strings containing the page of each image file.
    
    Returns:
    A numpy array whose elements are numpy arrays.  Each element array
    is the set of Haralick features for a particular image. 
    """
    print("Getting Haralick features...")
    features = []
    
    for image in images:
        image = mh.imread(image, as_grey=True).astype(np.uint8)
        # mahotas.features.haralick returns features in 4 directions. 
        features.append(mh.features.haralick(image).mean(0))
    print("Done getting standard Haralick features.")
    return np.array(features)
   
    
    
def get_visual_words(images, k):
    """
    Given a list of image files, finds local features using SURF.
    The local features are then partitioned into k clusters,
    where each cluster represents a visual word.  We then
    extract a bag of visual words for each image, which
    represents the feature set we can feed into a
    multinomial logistic classifier.
    
    Keyword arguments:
    images (list) - a list of strings representing the paths of training images
    
    k (int) - the number of visual words to create.  This should be between 256-1024,
    depending on the number of images used.  
        
    Returns:
    A numpy array of numpy arrays.  Each element array consists
    of the bag of visual words for a particular image.    
    """
    print("Getting surf descriptors...")
    alldescriptors = []
    
    for image in images:
        image = mh.imread(image, as_grey=True)
        image = image.astype(np.uint8)
        alldescriptors.append(surf.surf(image, descriptor_only=True))
        # alldescriptors is a list of numpy arrays
    
    # use a smaller sample of descriptors for speed
    # first get all descriptors into a single array,
    # then take every 32nd vector.
    concatenated = np.concatenate(alldescriptors)
    concatenated = concatenated[::34]
    print('Done getting all surf descriptors.')
    print('Creating visual words...')
    
    km = KMeans(k)
    km.fit(concatenated)
    surf_features = []
    
    for descriptor in alldescriptors:
        predicted = km.predict(descriptor)
        # for each image, predicted is an array of 
        # integers representing to which cluster each
        # descriptor of said image belongs.
        # [np.sum(predicted == i) for i in xrange(k)] is an array 
        # of length k that represents the bag of visual words for the image
        surf_features.append(
            np.array([np.sum(predicted == i) for i in xrange(k)])
            )
    
    visual_words = np.array(surf_features)
    print("Done creating the bags of visual words.")
    return visual_words

def get_tas_features(images):
    tas_features = []
    for image in images:
        image = mh.imread(image, as_grey=True)
        image = image.astype(np.uint8)
        tas_features.append(mh.features.tas(image))
    
    print("Done getting TAS features.")
    return np.array(tas_features)
        

def log_classify(feature_dict, labels):
    """
    Given a set of features in feature_dict, log_classify
    trains a logistic regression classifier using all possible subsets of
    features from feature_dict.  Each classifier is scored using k-fold
    cross-validation, and the set of features that gives the best
    performance is listed. 
    
    Keyword Arguments:
    feature_dict (dict) - a dictionary whose (key,value) pairs satisfy:
    key=a string representing a feature name
    value=numpy array that contains the features for eace image
        
    Returns:
    A numpy array consisting of the combination of features that 
    resulted in the highest cross-validation score.  
    Ready to use for fitting an sklearn classifier.
    
    """
    
    # Feature selection routine
    selection = [] 
    best_score = 0
    best_features = np.array([])
    
    # collect all the scores and feature names for graphing purposes
    all_scores = []
    names = []
    
    for l in range(len(feature_dict)):
        # get all subsets from the collection of features with length l+1
        combinations = itertools.combinations(feature_dict.keys(), l+1)
      
        for choice in combinations:
            # choice is a tuple of feature arrays
            temp_list = [feature_dict[feature] for feature in choice]
            # combine the subset of features into one array
            feature_array = np.hstack(temp_list)
            
            score = cross_validation.cross_val_score(
                LogisticRegression(), 
                feature_array, 
                labels, 
                cv=5
                ).mean()
            
            all_scores.append(score)
            names.append(str(choice))
            
            print(
                'Accuracy (5 fold x-val) with log. regression',
                list(choice),
                ': %s%%' % (0.1* round(1000*score))
                )
            
            if score > best_score:
                best_score = score
                #TODO get rid of selection?
                selection = list(choice)
                best_features = feature_array
    
    # get ready for graphing
    sorted_indicies = np.array(all_scores).argsort()
    min_index = sorted_indicies[0]
    sorted_scores = [all_scores[i] for i in sorted_indicies]
    sorted_names = [names[i] for i in sorted_indicies]

    # plotting
    #style.use('ggplot')
    fig = plt.figure()
    fig.suptitle('Logistic Regression', fontsize=20)
    plt.plot(range(len(sorted_scores)),100*np.array(sorted_scores), 'k-', lw=8)
    plt.plot(range(len(sorted_scores)),100*np.array(sorted_scores), 
        'o', mec='#825882', mew=12, mfc='#588282')
    plt.xlim(-.5,2.5)
    plt.ylim(all_scores[min_index]*90, best_score*110)
    plt.xticks(range(len(names)+1), names)
    plt.ylabel('Accuracy (%)')
    plt.savefig('img_classifying_graph_log.png')
    
    # return the set of features with the best performance
    return best_features
    

def svm_classify(std_features, surf_features, labels):
    score_std = cross_validation.cross_val_score(svm.SVC(), std_features, labels, cv=5)
    print('Accuracy (5 fold x-val) with svm [std features]: %s%%' % (0.1* round(1000*score_std.mean())))
    
    # do logistic regression with SURF features
    print('predicting...')
    scoreSURFlr = cross_validation.cross_val_score(
            svm.SVC(), surf_features, 
            labels, cv=5).mean()
    print('Accuracy (5 fold x-val) with svm [SURF features]: %s%%' % (0.1* round(1000*scoreSURFlr.mean())))
    
    # do logistic regression on the combined features
    print('Performing log. regression using combined features...')
    allfeatures = np.hstack([surf_features, std_features])
    score_combined = cross_validation.cross_val_score(svm.SVC(), allfeatures, labels, cv=5).mean()
    print('Accuracy (5 fold x-val) with svm [All features]: %s%%' % (0.1* round(1000*score_combined.mean())))
    
    # plotting
    #style.use('ggplot')
    fig = plt.figure()
    fig.suptitle('SVM', fontsize=20)
    plt.plot([0,1,2],100*np.array([score_std.mean(), scoreSURFlr, score_combined]), 'k-', lw=8)
    plt.plot([0,1,2],100*np.array(
        [score_std.mean(), scoreSURFlr, score_combined]), 
        'o', mec='#cccccc', mew=12, mfc='white')
    plt.xlim(-.5,2.5)
    plt.ylim(score_std.mean()*90., score_combined*110)
    plt.xticks([0,1,2], ["Standard", "SURF", "Combined"])
    plt.ylabel('Accuracy (%)')
    plt.savefig('img_classifying_graph_svm.png')

def tag_suggestion(image, image_tags, classes, classifier):
    """
    Keyword arguments:
    image - a png or jpg file
    image_tags (list) - a list of strings, each string a tag for the given image
    classes (list) - the set of images we have classified
    classifier (sklearn object) - a trained classifier
    """
    # read in the image using mahotas
    image = mh.imread(image, as_grey=True).astype(np.uint8)
    prediction = classifier.predict(image) 
    
def main():
    classes = [
        'chimp',
        'corvette',
        'tokyo',
        'goldengatebridge'
        ]
    
    images, labels = get_labels(classes)
    std_features = get_standard_features(images)
    
    k = 256
    surf_features = get_visual_words(images, k)
    tas_features = get_tas_features(images)
    
    feature_dict = {
        'Std': std_features,
        'SURF': surf_features,
        'TAS': tas_features
        #'Zernike': zernike_features
        }
        
    best_features = log_classify(feature_dict, labels)
    classifier = LogisticRegression() 
    classifier.fit(best_features, labels)
    
        
if __name__  == "__main__":
    main()
    
    
    
    
