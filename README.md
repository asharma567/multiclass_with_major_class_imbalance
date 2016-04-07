# multiclass_with_major_class_imbalance
This was a very interesting problem that I just had to tackle. Aside from the usual ML pipeline stuff: preprocessing, feature engineering, model evaluation, etc. This problem had the following key obstacles--
- An infinitely large dataset
- major class imbalance
- multilabel model evaluation

I make a fair attempt to solve for these issues in the analysis notebook posted here. Write-up's below.

__Classification Report)__
```
                             precision    recall  f1-score   support

             AndroidBrowser       0.37      0.32      0.34       188
                     Chrome       0.52      0.58      0.55       670
                  ChromeiOS       0.33      1.00      0.50         3
                    Firefox       0.42      0.11      0.17       123
                  GoogleBot       0.00      0.00      0.00         2
Microsoft Internet Explorer       0.54      0.66      0.59       348
                      Opera       0.75      0.28      0.41        32
                     Safari       0.14      0.01      0.02        95
                  UCBrowser       0.00      0.00      0.00         4
                    Unknown       0.49      0.56      0.52       527
                       misc       0.00      0.00      0.00         8

                avg / total       0.48      0.50      0.47      2000
```

__Using one multilabel model to handle the predictions of all the classes was a disaster in terms of performance which why mentioned in the next steps I would use a two-layer model approach.__

__Assumptions made--__

The user agent string is where I extracted the labels from; hence, I only used the UA string to make labels and not for anything else.

__Bottlenecks I came across--__
- Class imbalance
- Evaluation metric
- Only working with a small slice of the data

### Next steps

__EDA & Feature Engineering)__

One approach which came to mind was to use TSNE to look for groups within the data. If there were easily discernible groups then using a technique to the likes of K-means to cluster and classify new data points would make sense. Perhaps in combination with a silhouette score which could be later used as a feature. 

Computing the local time: By using the current time and the user's timezone we could compute the local time of the users at the time of click. From that we can extract hour of day, day of week, etc. I would assume there's certainly signal in this feature.

Apple users: One thing I suspect is most mac/apple users tend to be loyal to their products i.e. they're more likely to use Safari and other Apple products like iPhone, iPad, etc. Perhaps we can leverage each users global hash in this case or simply glean something from the OS they're using. Still noodling on ways to use this as a feature (or filter).

URL: I couldn't make use of the long url, but I feel there is a way and would've liked to spend more time looking into it.

__Models & Grid Search)__

Creating two-layers of classifiers: First layer would be to use an ensemble of binary classifiers to identify the majority class - 'Chrome' or not. This should serve a good initial way to even out the class imbalance. For the second layer, use an ensemble of multi-label models. 

Ensembling the models: For the first layer I would create a type of voting mechanism e.g. suppose I've fitted three binary classifiers to predict whether a data point is 'Chrome' or not. All the models would vote on the class of the new data point then the model's 'vote weight' would be a function of it's score e.g. roc_auc. Second layer would combine the top k results for a couple of multi-label classifiers e.g. top 5 highest probabilities for browser types and then a mechanism to pick the most probable class. Multi-label classifiers should perform better at this point since the classes are more balanced.

The specific models I would consider would be both linear and non-linear. Boosted trees in particular, Logistic regression (one versus all) & SVM (with linear kernal) as well. Though I would prepare the data differently with the linear models (e.g. scaling & k-1 dummy variable trap). It might also be worth trying to a dimensionality reduction technique e.g. PCA (just on the linear models) and examining the results.

After settling on an appropriate set of classifiers, I would grid search each model to their optimal parameters then compared the results. Just as an example, from past problems I know setting Logistic Regression's class_weight parameter to 'auto' does well for imbalanced classes. 

__Class imbalance)__

Perhaps combining the classes to a smaller set would be my first approach though, I would need to understand the business use-case before I started culling away minor class labels. The question would be, could we just live with the most popular names?

I think the two-layer modeling approach should sufficiently address this issue for the aforementioned reasons but there are also traditional ways to consider: i) undersampling ii) oversampling (using Noisy PCA, SMOTE, or K-modes) iii) sample weighting. 

__Bigger data)__

For the purposes of rapid prototyping I worked on a small set of the data for the take-home. Iterating through the problem using the entire dataset once the ML pipeline is properly prototyped would be a logically next step.

__Evaluation metric)__

Accuracy isn't a good metric for evaluating classifiers in general and even more so for multi-label problems. The primary reason being that accuracy is a function of a predefined threshold. There are metrics I would consider more reliable to the likes of Roc Auc or nDCG. Using f1 as a temporary fix might also make sense. I would certainly need to understand the business use-case a bit better before deciding e.g. we could optimizing for precision versus recall, etc.
