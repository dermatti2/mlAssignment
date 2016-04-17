---
title: "Machine Learning Assignment"
author: "Matthias Hellwig"
date: "April 17, 2016"
output: 
 html_document: 
    number_sections: yes
---

<blockquote>
 This report shows our solution to the Machine Learning Class of the Coursera class. We clean the given data by removing variables that contain not enough observations and random sampling from time windows. For the course prediction quiz submission we employ gbm (i.e. Boosting). We estimate our out of sample error at 81\%, whereas our prediction assignment accuracy is 100\%. 
</blockquote>

<h2>Reproducability</h2>
If you want to reproduce my results, clone my repository, set your working directory to it, and load the file <tt>ml_assignment_prediction.r</tt>. We have put this file also in the appendix, see below, so that you can see to which steps we refer in the following.

<h2>Data Preprocessing</h2>

<h3>Basic Observations</h3>

We have done an explorative analysis "by hand", i.e. we have imported the csv file to excel. The basic observations here are
<ul>
<li>many variables are not set (empty values or N/As)
<li>there are obviously time windows which we conclude from the names and the data of the variables <tt>raw_timestamp_part_1,	raw_timestamp_part 2, cvt_timestamp,	new_window,	num_window</tt></li>
</ul>

<h3>Selections of Observations</h3>

Since we have time windows in the data we have devised random subsampling to sample one observation per time window for our training data. The underlying idea here is of course that the data in one time window are very similar and we thus need only one observation per window. This should avoid overfitting and reduce running times of the algorithms. In fact we have tried building models without random sampling but the algorithms did not stop within several minutes. Our proceeding melts down the entire training sheet to 853 representative observations only giving reasonable running times (<3 mins). In file <tt>ml_assignment_prediction.r</tt> this random subsampling is done in Step 2) (see appendix).

<h3>Selection of Variables</h3>
<p> 
In order to deal with the above two observations, we select by hand the variables that are in neither one of the above mentioned sets. We further do not include the variable <tt>user_name</tt> since an adversary may deceive us by randomizing names and we want to make predictions from the "real" training data only (i.e. the data of physical activity). The exact list of variables is given by the varible <tt>variables</tt> in file <tt>ml_assignment_prediction.r</tt>. We have 53 variables in total, including the outcome <tt>classe</tt>. The subset selection takes places in Step 4)
</p>

<h3>Final Preprocessing Steps</h3>

We convert our sampled data into a data frame for which we have to conversions as R interprets our values as factor (Steps 3,5,6).  


<h2>Training</h2>

Our R-script is written so that is has two modes. With <tt>predictionMode = T</tt> it is configured to solve the course prediction quiz, with <tt>predictionMode = F</tt> it evaluates the model it has built. Depending on the predictionMode different sets are taken for variables <tt>trainingData</tt> and <tt>testingData</tt>.

For <tt>predicitionMode =T</tt> the entire (cleaned) training data set is taken for training the model. For <tt>predicitionMode =F</tt> (evaluation), we split the training set into a training and testing set and (80\% go into training, 20\% into testing). The training is done in Step 8. We have tried random forests and boosting with principical component analysis and without. See below why we have picked which variant for the prediction quiz.


<h2>Estimation of Out of Sample Error</h2>
For <tt>predictionMode = F</tt> our script measured how accurate our predictions are (in Step 10). This is done by comparing the predictions of variable <tt>classe</tt> on the testing set to its real value on the same data set. We have evaluated random forests with and without principal component analysis.  The different calls to train were

<ul>
<li> <pre>fit <- train( trainingData$classe  ~ ., method="gbm",  data=trainingData )</pre> </li>
<li> <pre>fit <- train( trainingData$classe  ~ ., method="gbm", preProcess="pca",  data=trainingData )/pre> </li>
<li> <pre>fit <- train( trainingData$classe  ~ ., method="rf",  data=trainingData )</pre> </li>
<li> <pre>fit <- train( trainingData$classe  ~ ., method="rf", preProcess="pca",  data=trainingData )</pre> </li>
</ul>

(Note that this cannot be seen from our script, which shows only the call with the best accuracy (in prediction mode)). The following tables gives the accuracies of the different methods (rounded to 1 decimal).

<table>
<tr><td>method&nbsp;</td><td> with pc &nbsp;</td><td> without pc&nbsp; </td></tr>
<tr><td>gbm</td><td>58.2\%</td><td>81.2\%</td></tr>
<tr><td>rf</td><td>64.7\%</td><td>79.4\%</td></tr>
</table>

Since boosting gives the best accuracy we choose boosting for the prediction quiz.


<h2>Solving Prediction assignment / Final Remarks </h2>
Our script outputs for <tt>predictionMode = T</tt> and boosting (gbm)

<pre>
 [1] "predicting"
 [1] B A B A A E D B A A B C B A E E A B B B
</pre>

This gives a result of 100\%. We believe that the explanation for the large discrepancy between the out of sample error rate and the predicition accurarcy explains as follows. We have estimated our out of sample error significant lower by the way the data were picked. We think that the instructors want all students to successfully acomplish the assignment and have chosen a non-representative subset of testing data that can be classified easily. We suppose for real data our predictions will be not that fair. However, since the result is already that high and the assignment took me that much time I abstain from using any further and advanced techniques like cross-validation to improve accuracy.

<h2>Appendix</h2>
For sake of completeness we list or R-script below. It is identical as the contents of file <tt>ml_assignment_prediction.r</tt>.

<pre>
library(rpart)
library(caret)

set.seed(1023)

predictionMode = T


print("reading data")
rawdata <- read.csv('data/pml-training.csv')

print("preprocessing")
# 1) split data by time windows
sets <- split(rawdata,rawdata$num_window)
# 2) pick per time window only one randomly sampled observation
train <- c()
for (i in (1:length(sets))) { 
  set <- as.data.frame(sets[i])
  numberOfObservations <- dim(set)[1]
  randomRow <- sample.int(numberOfObservations,1)
  train <- rbind(train,  as.matrix(set[randomRow,]))
}
# 3) conversion to data frame
train <- as.data.frame(train)
names(train) <- names(rawdata)


# 4) use only those predictors that have always values
variables <- c("roll_belt","pitch_belt","yaw_belt","total_accel_belt","gyros_belt_x","gyros_belt_y","gyros_belt_z","accel_belt_x","accel_belt_y","accel_belt_z","magnet_belt_x","magnet_belt_y","magnet_belt_z","roll_arm","pitch_arm","yaw_arm","total_accel_arm","gyros_arm_x","gyros_arm_y","gyros_arm_z","accel_arm_x","accel_arm_y","accel_arm_z","magnet_arm_x","magnet_arm_y","magnet_arm_z","roll_dumbbell","pitch_dumbbell","yaw_dumbbell","total_accel_dumbbell","gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z","accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z","magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z","roll_forearm","pitch_forearm","yaw_forearm","total_accel_forearm","gyros_forearm_x","gyros_forearm_y","gyros_forearm_z","accel_forearm_x","accel_forearm_y","accel_forearm_z","magnet_forearm_x","magnet_forearm_y","magnet_forearm_z")
trainClean <- train[,variables]

# 5) now convert to data frame with numeric values
trainCleanNumeric <- matrix(nrow = dim(trainClean)[1],ncol=dim(trainClean)[2])
for (i in seq(1,dim(trainClean)[1])) {
  for (j in seq(1,dim(trainClean)[2])) {
     trainCleanNumeric[i,j] <- as.numeric(as.character(trainClean[i,j]))
  }
}
# 6) include the outcome 
cleanedData <- cbind(train[,"classe"],as.data.frame(trainCleanNumeric))
names(cleanedData)<- c("classe",variables)
# now cleanedData contains the training data

# 7) setting training and testing data sets
if (predictionMode) {
  print("prediction mode")
  trainingData <- cleanedData
  testingData <- read.csv('data/pml-testing.csv')
} else {
 print("evaluation mode")
 print("subsetting")

 partition <- createDataPartition(y=cleanedData$classe,p=.8,list=F)
 trainingData <- cleanedData[partition,]
 testingData <- cleanedData[-partition,]
}

# 8) train the model
print("training model")

fit <- train( trainingData$classe  ~ ., method="gbm",  data=trainingData )

# 9) predict
print("predicting")
predictions <- predict(fit,newdata=testingData)

# 10) output predictions or evaluate
if (predictionMode) {
   print(predictions)
} else {
  print("evaluation")
  noCorrect <- sum(predictions== testingData$classe)
  noTotal <- length(testingData$classe)
  
  print(paste(noCorrect," correct predictions out of",noTotal," observations. Accuracy:",(noCorrect/noTotal)))
}
</pre>





