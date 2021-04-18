import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

def score_model(model, X, y):
    napovedi = model.predict(X)
    print(napovedi, "\n", y)
    return 1.0 - mean_absolute_error(napovedi, y)
    
def permutation_test_iteration(model_builder, X, y, Xtest, ytest):
    print("Starting new permutation test iteration ...")
    print("Building model ...")
    
    """
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    Xtest = np.nan_to_num(Xtest)
    ytest = np.nan_to_num(ytest)
    """
    
    model = model_builder(X, y)
    print("Model built. Scoring ...")
    acc = score_model(model, Xtest, ytest)
    print("Iteration accuracy:", acc)
    return acc
    
def permutation_test(model_builder, X, y, partitions = 10, iterations = 1):
    print("Starting permutation test ...")
    
    d = np.c_[X, y.T]
    np.random.shuffle(d)
    partitioned = np.array_split(d, partitions)
    accuracies = []
    for i in range(partitions):
        # divide all data into training and testing set
        print("Dividing data into training and testing set")
        train = np.concatenate([partitioned[k] for k in range(partitions) if not k == i])
        train_data = { 'X': train[:,:-1], 'y': np.squeeze(np.asarray( train[:,-1] )) }
        test_data = { 'X': partitioned[i][:,:-1], 'y': np.squeeze(np.asarray( partitioned[i][:,-1] )) }
        
        #print(train_data)
        #print(test_data)
        
        acc = permutation_test_iteration(model_builder, train_data["X"], train_data["y"], test_data["X"], test_data["y"])
        accuracies.append(acc)
    
    avg_acc = np.mean( accuracies )
    print("Prmutation test finished. Accuracy:", avg_acc)
    return avg_acc