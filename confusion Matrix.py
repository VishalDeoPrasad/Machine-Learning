import numpy as np
def rates_frm_mat(mat):
    tpr,fpr,tnr,fnr = None,None,None,None

    #Calculate true positives, true negatives, false positives and false negatives
    TN, FN, FP, TP = mat[0][0], mat[0][1], mat[1][0], mat[1][1]
    #print(TP, FP, FN, TN) 
    
    #Calculate true positive rate, false positive rate, true negative rate and false negative rate
    tpr = TP/(TP+FN)
    fpr = FP/(TN+FP)
    tnr = TN/(TN+FP)
    fnr = FN/(TP+FN)

    
    #return a NumPy array with elements tpr, fpr, tnr, fnr all of which are rounded off to two decimal places   
    return np.array([tpr,fpr,tnr,fnr]).round(2)

mat = [[850, 6], [50, 94]]
print(rates_frm_mat(mat))