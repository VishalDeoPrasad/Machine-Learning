def accuracy_metric(actual, predicted):
    accuracy = 0
    correct_pred = 0
    wrong_pred = 0
    # for i in range(len(actual)):
    #     if actual[i] == predicted[i]:
    #         correct_pred += 1
    # accuracy = correct_pred/len(actual)
    #[correct_pred := correct_pred + 1 if actual[i] == predicted[i] else wrong_pred := wrong_pred+1 for i in range(len(actural))]
    [correct_pred := correct_pred + 1 for i in range(len(actual)) if actual[i] == predicted[i]]
    accuracy = correct_pred/len(actual)

    return accuracy*100

actual = [1, 0, 2]
predicted = [1, 0, 1]
print(accuracy_metric(actual, predicted))
