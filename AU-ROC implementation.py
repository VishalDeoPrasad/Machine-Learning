import pandas as pd
import numpy as np
def au_roc(prob, labels):
    
    data = pd.DataFrame({"probab" : prob, "y" : labels})
    
    #storing all the threshold values sorted in descending order
    thr=  list(set(data['probab']))
    thr.sort(reverse = True)
    
    tpr_arr,fpr_arr = [],[]

    for i in thr:
        #adding new column y_pred based on probability score and threshold
        data['y_pred'] = data['probab'].apply(lambda y_score: 0 if y_score < i else 1)
        
        #calculate true negatives, false positives, false negatives and true positives
        tn = len(data[(data.y == 0) & (data.y_pred == 0)])
        fp = len(data[(data.y== 0) & (data.y_pred == 1)])
        fn = len(data[(data.y == 1) & (data.y_pred == 0)])
        tp = len(data[(data.y == 1) & (data.y_pred == 1)])
        
        #calculate true positive rate and false positive rate
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        
        tpr_arr.append(tpr)
        fpr_arr.append(fpr)
    
    #Calculating the area under curve using trapz function
    auroc = np.trapz(tpr_arr, fpr_arr)
    return auroc.round(2)

prob = [0.64, 0.01, 0.22, 0.44, 0.02, 0.64, 0.87, 0.  , 0.06, 0.8 ]
labels = [1, 0, 1, 1, 0, 0, 1, 1, 0, 0]
print(au_roc(prob, labels))

