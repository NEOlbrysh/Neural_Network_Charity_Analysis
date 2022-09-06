# Neural_Network_Charity_Analysis

Charity organization analysis using neural networks.

## Overview


The purpose of this project was to create a binary classifier, using a neural network, to attempt to predict whether applicant to a charity organization, Alphabet Soup, will be successful with their funding. The starting dataset consisted of a CSV of over 34,000 organizations that received funding from Alphabet Soup. The columns in the dataset included:

-	EIN and NAME — Identification columns
-	APPLICATION_TYPE — Alphabet Soup application type
-	AFFILIATION — Affiliated sector of industry
-	CLASSIFICATION — Government organization classification
-	USE_CASE — Use case for funding
-	ORGANIZATION — Organization type
-	STATUS — Active status
-	INCOME_AMT — Income classification
-	SPECIAL_CONSIDERATIONS — Special consideration for application
-	ASK_AMT — Funding amount requested
-	IS_SUCCESSFUL — Was the money used effectively


The libraries used were:

- Pandas
- Scikit-learn
- Tensorflow


The goal was to attempt to develop a model with over 75% accuracy in predicting success from the given data.


## Results

### Preprocessing

The first step was to examine and preprocess the provided dataset.

- The target variable was the IS_SUCCESSFUL column.
- The unnecessary data for the purposes of this model were the EIN and NAME columns.
- All other columns were considered potential features for the model. The next steps were to bin, encode, and scale the data.
- Any APPLICATION_TYPE with less than 1000 entries were binned into OTHER.
- Any CLASSIFICATION with less than 1000 entries were binned into OTHER.
- All "object" type columns were encoded using OneHotEncoder.
- The SPECIAL_CONSIDERATIONS_N column was dropped, as it was redundant to the SPECIAL_CONSIDERATIONS_Y column.
- All columns were then scaled using StandardScaler.


## Compiling, Training, and Evaluating


In the initial model I used:

-	Two layers -- one with 80 neurons, the second with 45 neurons -- providing me with 6,891 total and trainable parameters.
-	Both layers used 'relu' activation functions.
-	The output layer used 'sigmoid' activation function. Unfortunately, I was only able to achieve 72.6% accuracy with this model.

I tried three more models to reach 75% accuracy. In my subsequent attempts I attempted:

-	Binning INCOME_AMT values greater than $5 million into a '5M+' bin.
-	Adding a third hidden layer.
-	Increasing the total number of trainable parameters to as high as 9,411.
-	Increasing training epochs from 100 to 150, then as high as 300.
-	Trying out both the 'adamax' and 'nadam' optimizers when compiling the model.
-	Using 'tanh' activation functions on the hidden layers.
-	Un-binning certain values by lowering the threshold from 1000 values to 700 values on both APPLICATION_TYPE and CLASSIFICATION.

Across all four of my attempts, I never managed to raise my models' accuracy above 73%.


## Summary
Neural networks are complex machines. There are lots of moving part to keep track of. There are also lots of "black boxes" which make analyzing both the processes and results into a hazy, vague endeavor. I'm honestly not sure what would help improve accuracy on these models, except for a lot of brute-force experimentation. Increasing parameters or training epochs don't seem to be that effective. Perhaps attempting some other methods, such as random forests or SVM might yield better results. Maybe there are outliers which I just was unable to pin-point in my examinations. That said, 73% accuracy isn't that bad in the world of finance. 
