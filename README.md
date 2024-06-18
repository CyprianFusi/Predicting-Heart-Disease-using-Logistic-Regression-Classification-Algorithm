# Predicting-Heart-Disease-using-Logistic-Regression-Classification-Algorithm
The model's CAP curve shows we are having **`100%`**! This means it is capable of correctly predicting **100%** of patients with a heart disease after processing 50% of the data. The model's performance is **"Too Good to be True"**! However, with **`Train accuracy = 86%`** and **`Test accuracy = 82%`**, there is no visible sign of overfitting.

### Observations
* Train accuracy: `86%`
* Test accuracy: `80%`
* Precision is:  `86%`
* The model gets confused in 11 cases
    * The model classified 3 **healthy people** as having a **heart disease** (**`False Positive` or `Type I error`**)
    * The model classified 8 **heart disease patients** as **healthy people** (**`False Negative` or `Type II error`**)
    
* The number of wrong predictions remains 12. However, **the number of FN has dropped from 8 to 7**. Which is good news for model's **precision** score.
* With a test accuracy of `80%`, the model is not doing badly, but it's not good enough for the application of detecting heart disease
* The model was trained and tested with just **`303`**! This is certainly not enough data to train a model for such an application. More data is needed to train and deploy the model in production.
    
#### Important Note:
With **Type I error**, the healthy individual would have to through treatment. If it were wrong cancer diagnose the individual would have to undergo **chemotherapy** with the accompanying side effects!

With **Type II error**, the sick individual would probably **NOT get any treatment and might even die from the illness!!**

The question is often asked, which of these errors is better? Well, it depends on the application. In the case of detecting illness, **the goal is to make sure that no illness go undetected!** The consequences of missing to detect an illness is more serious than the side effects incurred from administrying treatment on a healthy individual.

To improve your belief of having a disease or not having one, **it's highly recommended to go for a second test** in order to double sure about the likelihood or having the disease or free from it. There is, however, a caveat here! The second test should be from the same test provider (not the same testing clinic). This is because **it's very unlikely for the same test to make an error on same patient twice!** If it would. well...
