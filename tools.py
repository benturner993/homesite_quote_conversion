import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error, roc_curve, auc
# plt.rcParams["figure.figsize"] = (10,10)

def plot_one_ways(i, diff_band, exposure, obs, target_name, width=0.5):
    
    """
    function to create one way plots
    - diff_band: the x co-ordinates for plots
    - exposure: number of rows per diff_band grouping
    - width: width of columns predefined as 0.5
    - target_name: response to plot as line
    """
    
    plt.bar(diff_band, exposure, width, color='gold', label='Diff', edgecolor='k')
    plt.xticks(rotation=90)
    plt.ylim(0, max(exposure)*3)
    plt.ylabel('Exposure')
    plt.xlabel(f'{i}')
    plt.title(f'One Way Plot of {i}')

    # Line plot
    axes2 = plt.twinx()
    axes2.plot(diff_band, obs, color='fuchsia', marker="s", markeredgecolor='black', label='Actual')
    axes2.set_ylabel(target_name)

    # legend and settings
    plt.legend(loc="upper left")

    plt.show()
    
def remove_outliers(df, col, constraint=1.5):
    
    """
    function to detect outliers for any given df, column and constraint (defaulted to 1.5)
    returns new column with _CL suffix to indicate outliers have been removed
    """

    # find quartiles
    q1 = pd.DataFrame(df[col]).quantile(0.25)[0]
    q3 = pd.DataFrame(df[col]).quantile(0.75)[0]
    
    # find interquartile range
    iqr = q3 - q1 
    
    # determine upper and lower bounds
    lower_bound = q1 - (constraint*iqr)
    upper_bound = q3 + (constraint*iqr)
    print(lower_bound, upper_bound)
    
    # determine upper and lower bounds
    df[col+'_CL']=np.where(df[col]<=lower_bound,lower_bound,df[col])
    df[col+'_CL']=np.where(df[col]>=upper_bound,upper_bound,df[col+'_CL'])
    print(col+'_CL created...')
    
def training_validation_subset(df):
    ''' function to create training and validation subsets
        chosen this methodology as a method to replicate in the future '''

    training_df = df.sample(frac=0.7)
    print('Training dataset rows:\t', training_df.shape[0])

    validation_df = pd.concat([df, training_df]).drop_duplicates(keep=False)
    print('Validation dataset rows:\t', validation_df.shape[0])

    return training_df, validation_df

def downsample_func(df, fraction=1):

    ''' function to downsample dataset with replacement for any given input df and specified fraction
        default fraction set to 1 '''

    # randomly select data with replacement
    df=df.sample(frac = fraction, replace = True)
    return df

def downsampled_results(df, pred1, act, n):

    ''' function which creates n downsampled datasets and calculates the mae '''

    original_mae=mean_absolute_error(df[act], df[pred1])
    print('Original: ', original_mae)
    
    pred1_maes=[]
    
    for i in range(1, n):
        
        # create a downsampled dataframe
        down_df=downsample_func(df)
                    
        # calculate mae on sample
        pred1_mae=mean_absolute_error(df[act], df[pred1])
        pred1_maes.append(pred1_mae)
                    
    # convert list into an array
    pred1_a=np.array(pred1_maes)
    
    # calculate mae and confidence intervals
    pred1_p5 = round(np.percentile(pred1_a, 5),3)
    pred1_p50 = round(np.percentile(pred1_a, 50),3)
    pred1_p95 = round(np.percentile(pred1_a, 95),3)
    print('Lower Bound: ', pred1_p5)
    print('Median: ', pred1_p50)
    print('Upper Bound: ', pred1_p95)
    
def model_performance_metrics(y_train, train_pred, train_prob):

    ''' function to return the model performance metrics '''

    lst = [0] * len(y_train)
    tn, fp, fn, tp = confusion_matrix(train_pred, y_train).ravel()
    print(f'True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}')
    print(f'Null Accuracy Score: {accuracy_score(y_train, pd.Series(lst))}')
    print(f'Accuracy Score: {accuracy_score(y_train, train_pred)}')
    print(f'Precision Score: {precision_score(y_train, train_pred, average=None)}')
    print(f'Recall Score: {recall_score(y_train, train_pred, average=None)}')
    print(f'F1 Score: {f1_score(y_train, train_pred, average=None)}')
    print(f'AUROC: {roc_auc_score(y_train, train_prob)}')
    print(f'MCC: {matthews_corrcoef(y_train, train_pred)} \n')

def optimal_cutoff(act, pred):
    
    """ find the optimal probability cutoff point for a classification model 
        credit to https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    """
    
    fpr, tpr, threshold = roc_curve(act, pred)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'value' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['value']) 

def add_features(data_in):

    data_out = data_in.drop(['QuoteNumber'], axis=1)

    # remove quote conversion flag if it exists
    if 'QuoteConversion_Flag' in data_out.columns:
        data_out = data_out.drop(['QuoteConversion_Flag'], axis=1)

    # binary day of week features
    week_dict = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday': 6}
    for day in week_dict:
        weekday = lambda x: x.dayofweek
        data_out[day] = (data_in['Original_Quote_Date'].apply(weekday)==week_dict[day]).astype(int)

    return data_out

#Generic function for making a classification model and accessing the performance. 
# From AnalyticsVidhya tutorial
def classification_model(model, data, predictors, outcome):
    
    #Fit the model:
    model.fit(data[predictors],data[outcome])
  
    #Make predictions on training set:
    predictions = model.predict(data[predictors])
  
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print("Training Accuracy : %s" % "{0:.3%}".format(accuracy))

    #Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=5)
    error = []
    for train, test in kf.split(data[predictors]):
        # Filter training data
        train_predictors = (data[predictors].iloc[train,:])
    
        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]
    
        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
    
        #Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    
        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
    #Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors],data[outcome])