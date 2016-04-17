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

fit <- train( trainingData$classe  ~ ., method="gbm", data=trainingData )

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





