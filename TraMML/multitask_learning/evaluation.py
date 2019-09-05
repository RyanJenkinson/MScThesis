# Contains some simple evaluation functions to assess the accuracy of our models.
import numpy as np
from scipy.special import softmax
from sklearn import metrics
from data_processing import NERProcessor
TASK_EVALUATION_SETTINGS = {'SemEval': {'num_sent_categories': 5},
                            'Sentihood': {'num_sent_categories': 3}}
def Macro_PRF(y_true, y_preds, task_name):
    """
    Calculate the precision, recall and f1 score for the SemEval and Sentihood dataset

    Parameters
    ----------
    y_true : (numpy) array
        Array of true y values
    y_preds : (numpy) array
        Array of predicted y values

    Returns
    -------
    tuple
        precision, recall, f1_score
    """
    num_sent_categories = TASK_EVALUATION_SETTINGS[task_name.split("_")[0]]['num_sent_categories']

    pred_sent_count = 0
    actual_sent_count = 0
    intersection_sent_count = 0

    # For each unique text, loop over each possible sentiment and
    # append predicted, actual and intersected sents
    for text_id in range(len(y_preds)//num_sent_categories):
        pred_sents = set()
        actual_sents = set()
        for sent_id in range(num_sent_categories):
            current_idx = num_sent_categories*text_id + sent_id
            if y_preds[current_idx] != -1:  # i.e if not "None"
                pred_sents.add(sent_id)
            if y_true[current_idx] != -1:  # i.e if not "None"
                actual_sents.add(sent_id)
        if len(actual_sents) == 0:
            continue
        intersected_sents = actual_sents.intersection(pred_sents)
        pred_sent_count += len(pred_sents)
        actual_sent_count += len(actual_sents)
        intersection_sent_count += len(intersected_sents)

    # Calculate the precision, recall and f1_score
    precision = intersection_sent_count / pred_sent_count
    recall = intersection_sent_count / actual_sent_count
    f1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1_score

def Sentihood_strict_acc(y_true, y_pred):
    """
    Calculate "Strict Accuracy" of aspect detection task of Sentihood.
    There are 4 types of category we are considering in the dataset:
    ['General', 'Price', 'Safety', 'Transit-location']
    """
    total_cases = int(len(y_true)/4)
    true_cases = 0
    for i in range(total_cases):
        if all(y_pred[(4*i):(4*i+3)] == y_true[(4*i):(4*i+3)]):
            true_cases += 1
    aspect_strict_acc = true_cases/total_cases

    return aspect_strict_acc


def SemEval_acc(y_true, scores, n_classes=4):
    """
    Calculate the n_class accuracy for the SemEval Dataset

    Parameters
    ----------
    y_true : (numpy) array
        Array of true y values
    y_preds : (numpy) array
        Array of predicted y values as outputted by our model
    scores : (numpy) array
        a 2D array where each row is a vector of scores/logits for each possible label
    n_classes : int, optional
        Number of classes to calculate the accuracy to, by default 4

    Returns
    -------
    float
        The n_class accuracy

    Raises
    ------
    ValueError
        if n_classes is not in [2,3,4]
    """
    if n_classes not in [2, 3, 4]:
        raise ValueError("The number of classes for SemEval accuracy must be in [2, 3, 4]!")

    # For each true label < n_classes, calculate the best prediction out of those classes
    # and return the accuracy
    total_correct, total = 0, 0
    for i in range(len(y_true)):
        if y_true[i] >= n_classes or y_true[i] == -1:  # If the true value is a null (-1) category or outside the n_classes range then skip example
            continue
        total += 1
        # Get the prediction for the number of classes we are calculating the accuracy to
        pred = np.argmax(scores[i][:n_classes])
        if pred == y_true[i]:
            total_correct += 1
    acc = total_correct / total
    return acc

def Sentihood_AUC_Acc(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of Sentihood.
    Calculate "Acc" of sentiment classification task of Sentihood.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[]]
    aspect_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i] == -1:
            aspect_y_true.append(1)  # Let probability of "None" be represented by a 1
        else:
            aspect_y_true.append(0)
        none_prob = softmax(score[i])[-1] # probability of "None"
        aspect_y_score.append(none_prob)
        aspect_y_trues[i%4].append(aspect_y_true[-1])
        aspect_y_scores[i%4].append(aspect_y_score[-1])

    aspect_auc = []
    for i in range(4):
        aspect_auc.append(metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i]))
    aspect_Macro_AUC = np.mean(aspect_auc)

    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_score=[]
    sentiment_y_trues=[[],[],[],[]]
    sentiment_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i] != -1:
            sentiment_y_true.append(abs(y_true[i]-1)) # "Postive":0, "Negative":1
            neg_prob = softmax(score[i])[0]/(softmax(score[i])[0]+softmax(score[i])[1])  # probability of "Negative"
            sentiment_y_score.append(neg_prob)
            if neg_prob>0.5:
                sentiment_y_pred.append(1) # "Negative": 1
            else:
                sentiment_y_pred.append(0)
            sentiment_y_trues[i%4].append(sentiment_y_true[-1])
            sentiment_y_scores[i%4].append(sentiment_y_score[-1])

    sentiment_auc=[]
    for i in range(4):
        sentiment_auc.append(metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i]))
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC

def NER_eval(logits, label_ids):
    """
    NER Evaluation from the logits and true_label_ids
    get the y_true and y_pred for a given batch

    Parameters
    ----------
    logits : array
        Array of logits for the batch output
    label_ids : array
        Array of the true label_ids for a given batch

    Returns
    -------
    tuple
        y_true, y_pred
    """
    # Define a label_map so that the code is more readable. Don't predict the CLS or SEP labels so start label map from 2 onwards
    label_list = NERProcessor("../../../datascience-projects/internal/multitask_learning/processed_data/NER").get_labels()
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[i] = label

    # Create a list of y_true and y_pred
    y_true_all = []
    y_pred_all = []
    #label_ids is a list of lists
    for example_idx, example_true_label_id_list in enumerate(label_ids):
        y_true = []
        y_pred = []
        for label_num, label_id in enumerate(example_true_label_id_list):
            if label_id == -1:  # i.e not a valid label (e.g start and end/padding of the sentence)
                continue
            else:
                y_true.append(label_map[label_id])
                y_pred.append(label_map[np.argmax(logits[example_idx][label_num])])
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
    return y_true_all, y_pred_all
