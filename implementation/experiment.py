import warnings
import random
import pandas as pd
import numpy as np
import statistics
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

def get_PPC_one_way(file, criteria):
    df = pd.read_csv(file)
    
    header = "range, lower, upper,  precision, recall, F1, accuracy, coverage"
    print(header)

    for i in range(1, 51):
        inc = i/100
        upper = 0.5 + inc
        lower = 0.5 - inc
        condition = (df['bayesfactor'].str.contains('inf') | df['bayesfactor'].str.contains('NaNe-Inf'))
        lower_criteria = df[criteria] >= lower
        upper_criteria = df[criteria] <= upper
        
        #final = df.loc[condition & lower_criteria & upper_criteria]
        final = df.loc[lower_criteria & upper_criteria]
        
        coverage = len(final.sentence_source)/len(df.index)
        
        print( 
            '50\% $\pm$ ' + str(int(round(100 * inc, 0))) + "\%", ",",
            '%.0f' % (100 * lower) + "\%", ",",
            '%.0f' % (100 * upper) + "\%", ",",
            '%.2f' % (100 * precision_score(final.label, final.predicted)) + "\%", ",",
            '%.2f' % (100 * recall_score(final.label, final.predicted)) + "\%", ",",
            '%.2f' % (100 * f1_score(final.label, final.predicted)) + "\%", ",",
            '%.2f' % (100 * accuracy_score(final.label, final.predicted)) + "\%", ",",
            '%.2f' % (100 * coverage) + "\%"
        )

def get_PPC_complete(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    header = "range, lower, upper,  precision, recall, F1, accuracy, coverage"
    print(header)

    for i in range(1, 51):
        inc = i/100
        upper = 0.5 + inc
        lower = 0.5 - inc
        
        df1_condition = (df1['bayesfactor'].str.contains('inf') | df1['bayesfactor'].str.contains('NaNe-Inf'))
        df1_lower = df1['max_test'] >= lower
        df1_upper = df1['max_test'] <= upper
        #df1_final = df1.loc[df1_condition & df1_lower & df1_upper]
        df1_final = df1.loc[df1_lower & df1_upper]
        
        df2_condition = (df2['bayesfactor'].str.contains('inf') | df2['bayesfactor'].str.contains('NaNe-Inf'))
        df2_lower = df2['min_test'] >= lower
        df2_upper = df2['min_test'] <= upper
        #df2_final = df2.loc[df2_condition & df2_lower & df2_upper]
        df2_final = df2.loc[df2_lower & df2_upper]
        
        df3 = pd.concat([df1_final[["sentence_source", "label", "predicted"]], df2_final[["sentence_source", "label", "predicted"]]]).drop_duplicates(subset=['sentence_source'])

        coverage = len(df3.sentence_source)/len(df1.index)

        print( 
            '50\% $\pm$ ' + str(int(round(100 * inc, 0))) + "\%", ",",
            '%.0f' % (100 * lower) + "\%", ",",
            '%.0f' % (100 * upper) + "\%", ",",
            '%.2f' % (100 * precision_score(df3.label, df3.predicted)) + "\%", ",",
            '%.2f' % (100 * recall_score(df3.label, df3.predicted)) + "\%", ",",
            '%.2f' % (100 * f1_score(df3.label, df3.predicted)) + "\%", ",",
            '%.2f' % (100 * accuracy_score(df3.label, df3.predicted)) + "\%", ",",
            '%.2f' % (100 * coverage) + "\%"
        )

def get_PPC_complete_pretty_print(file1, file2):
    x_axis = []
    f1 = []
    accuracy = []
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    header = "range, lower, upper,  precision, recall, F1, accuracy, coverage"
    print(header)

    for i in range(1, 51):
        inc = i/100
        upper = 0.5 + inc
        lower = 0.5 - inc
        
        df1_condition = (df1['bayesfactor'].str.contains('inf') | df1['bayesfactor'].str.contains('NaNe-Inf'))
        df1_lower = df1['max_test'] >= lower
        df1_upper = df1['max_test'] <= upper
        #df1_final = df1.loc[df1_condition & df1_lower & df1_upper]
        df1_final = df1.loc[df1_lower & df1_upper]
        
        df2_condition = (df2['bayesfactor'].str.contains('inf') | df2['bayesfactor'].str.contains('NaNe-Inf'))
        df2_lower = df2['min_test'] >= lower
        df2_upper = df2['min_test'] <= upper
        #df2_final = df2.loc[df2_condition & df2_lower & df2_upper]
        df2_final = df2.loc[df2_lower & df2_upper]
        
        df3 = pd.concat([df1_final[["sentence_source", "label", "predicted"]], df2_final[["sentence_source", "label", "predicted"]]]).drop_duplicates(subset=['sentence_source'])

        coverage = len(df3.sentence_source)/len(df1.index)

        x_axis.append('50% \u00B1 ' + str(int(round(100 * inc, 0))) + "%")
        f1.append(f1_score(df3.label, df3.predicted))
        accuracy.append(accuracy_score(df3.label, df3.predicted))

    plt.figure()
    plt.plot(x_axis, f1, 'r', label = 'F1 Score')
    plt.plot(x_axis, accuracy, 'b', label = 'Accuracy')
    plt.legend(loc="upper right", prop={'size': 12})
    plt.xticks([0, 9, 19, 29, 39, 49])
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylabel('Percentage', fontsize=12)
    plt.xlabel('Range', fontsize=12)
    #plt.title('Model Evaluation')
    plt.show()

def get_Model_Evaluation(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    header = "Range, Lower, Upper, Precision, Recall, F1, Accuracy, Coverage, Bayes Factor"
    print(header)

    for i in range(1, 51):
        inc = i/100
        upper = 0.5 + inc
        lower = 0.5 - inc
        
        df1_condition = (df1['bayesfactor'].str.contains('inf') | df1['bayesfactor'].str.contains('NaNe-Inf'))
        df1_lower = df1['max_test'] >= lower
        df1_upper = df1['max_test'] <= upper
        df1_likelihood = df1['likelihood_e1_arrow_e2'] == 0
        df1_final = df1.loc[(df1_condition & df1_lower & df1_upper & df1_likelihood) | ~df1_likelihood]
 
        df2_condition = (df2['bayesfactor'].str.contains('inf') | df2['bayesfactor'].str.contains('NaNe-Inf'))
        df2_lower = df2['min_test'] >= lower
        df2_upper = df2['min_test'] <= upper
        df2_likelihood = df2['likelihood_e2_arrow_e1'] == 0
        df2_final = df2.loc[(df2_condition & df2_lower & df2_upper & df2_likelihood) | ~df2_likelihood]
         
        df3 = pd.concat([df1_final[["sentence_source", "label", "predicted"]], df2_final[["sentence_source", "label", "predicted"]]]).drop_duplicates(subset=['sentence_source'])
        df3.to_csv("test_"+str(i)+".csv")
        
        coverage = len(df3.sentence_source)/len(df1.index)

        print( 
            '50\% $\pm$ ' + str(int(round(100 * inc, 0))) + "\%", ",",
            '%.0f' % (100 * lower) + "\%", ",",
            '%.0f' % (100 * upper) + "\%", ",",
            '%.2f' % (100 * precision_score(df3.label, df3.predicted)) + "\%", ",",
            '%.2f' % (100 * recall_score(df3.label, df3.predicted)) + "\%", ",",
            '%.2f' % (100 * f1_score(df3.label, df3.predicted)) + "\%", ",",
            '%.2f' % (100 * accuracy_score(df3.label, df3.predicted)) + "\%", ",",
            '%.2f' % (100 * coverage) + "\%", ",",
            'inf/NaNe-Inf',
        )

def get_Model_Evaluation_pretty_print(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    x_axis = []
    f1 = []
    accuracy = []
    
    header = "Range, Lower, Upper, Precision, Recall, F1, Accuracy, Coverage, Bayes Factor"
    print(header)

    for i in range(1, 51):
        inc = i/100
        upper = 0.5 + inc
        lower = 0.5 - inc
        
        df1_condition = (df1['bayesfactor'].str.contains('inf') | df1['bayesfactor'].str.contains('NaNe-Inf'))
        df1_lower = df1['max_test'] >= lower
        df1_upper = df1['max_test'] <= upper
        df1_likelihood = df1['likelihood_e1_arrow_e2'] == 0
        df1_final = df1.loc[df1_condition & df1_lower & df1_upper & df1_likelihood ]
        df1_final = df1.loc[df1_condition & df1_likelihood ]
        #df1_final = df1.loc[df1_lower & df1_upper]
        
        df2_condition = (df2['bayesfactor'].str.contains('inf') | df2['bayesfactor'].str.contains('NaNe-Inf'))
        df2_lower = df2['min_test'] >= lower
        df2_upper = df2['min_test'] <= upper
        df2_likelihood = df2['likelihood_e2_arrow_e1'] == 0
        df2_final = df2.loc[df2_condition & df2_lower & df2_upper & df2_likelihood]
        df2_final = df2.loc[df2_condition & df2_likelihood]
        #df2_final = df2.loc[df2_lower & df2_upper]
        
        df3 = pd.concat([df1_final[["sentence_source", "label", "predicted"]], df2_final[["sentence_source", "label", "predicted"]]]).drop_duplicates(subset=['sentence_source'])

        coverage = len(df3.sentence_source)/len(df1.index)

        x_axis.append('50% \u00B1 ' + str(int(round(100 * inc, 0))) + "%")
        f1.append(f1_score(df3.label, df3.predicted))
        accuracy.append(accuracy_score(df3.label, df3.predicted))
        
    plt.figure()
    plt.plot(x_axis, f1, 'r', label = 'F1 Score')
    plt.plot(x_axis, accuracy, 'b', label = 'Accuracy')
    plt.legend(loc="upper right", prop={'size': 12})
    plt.xticks([0, 9, 19, 29, 39, 49])
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylabel('Percentage', fontsize=12)
    plt.xlabel('Range', fontsize=12)
    #plt.title('Model Evaluation')
    plt.show()
        
def get_Model_Evaluation_single_entry(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    header = "range, lower, upper,  precision, recall, F1, accuracy, coverage"
    print(header)

    upper = 0.5 + 0.03
    lower = 0.5 - 0.03
    
    df1_condition = (df1['bayesfactor'].str.contains('inf') | df1['bayesfactor'].str.contains('NaNe-Inf'))
    df1_lower = df1['max_test'] >= lower
    df1_upper = df1['max_test'] <= upper
    df1_final = df1.loc[df1_condition & df1_lower & df1_upper]
    #df1_final = df1.loc[df1_lower & df1_upper]
        
    df2_condition = (df2['bayesfactor'].str.contains('inf') | df2['bayesfactor'].str.contains('NaNe-Inf'))
    df2_lower = df2['min_test'] >= lower
    df2_upper = df2['min_test'] <= upper
    df2_final = df2.loc[df2_condition & df2_lower & df2_upper]
    #df2_final = df2.loc[df2_lower & df2_upper]
    
    df3 = pd.concat([df1_final[["sentence_source", "label", "predicted", "bayesfactor"]], df2_final[["sentence_source", "label", "predicted", "bayesfactor"]]]).drop_duplicates(subset=['sentence_source'])

    #print(df3)
    
    coverage = len(df3.sentence_source)/len(df1.index)
    print( 
        '50\% $\pm$ ' + str(int(round(100 * 0.03, 0))) + "\%", ",",
        '%.0f' % (100 * lower) + "\%", ",",
        '%.0f' % (100 * upper) + "\%", ",",
        '%.2f' % (100 * precision_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * recall_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * f1_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * accuracy_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * coverage) + "\%"
    )
    
def get_performance_metrics_BF_on(file1, file2, i):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1_condition = (df1['bayesfactor'].str.contains('inf') | df1['bayesfactor'].str.contains('NaNe-Inf'))
    df1_likelihood = df1['likelihood_e1_arrow_e2'] == 0
    df1_final = df1.loc[(df1_condition & df1_likelihood) | ~df1_likelihood]
    
    df2_condition = (df2['bayesfactor'].str.contains('inf') | df2['bayesfactor'].str.contains('NaNe-Inf'))
    df2_likelihood = df2['likelihood_e2_arrow_e1'] == 0
    df2_final = df2.loc[(df2_condition & df2_likelihood) | ~df2_likelihood]
    
    df3 = pd.concat([df1_final[["sentence_source", "label", "predicted"]], df2_final[["sentence_source", "label", "predicted"]]]).drop_duplicates(subset=['sentence_source'])
    coverage = len(df3.sentence_source)/len(df1.index)
    print(
        i, ",",
        '%.2f' % (100 * precision_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * recall_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * f1_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * accuracy_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * coverage) + "\%"
    )
    return precision_score(df3.label, df3.predicted), recall_score(df3.label, df3.predicted), f1_score(df3.label, df3.predicted), accuracy_score(df3.label, df3.predicted), coverage
    
def get_performance_metrics_BF_off(file1, file2, i):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.concat([df1[["sentence_source", "label", "predicted"]], df2[["sentence_source", "label", "predicted"]]]).drop_duplicates(subset=['sentence_source'])
    coverage = len(df3.sentence_source)/len(df1.index)
    print(
        i, ",",
        '%.2f' % (100 * precision_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * recall_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * f1_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * accuracy_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * coverage) + "\%", ",",
        '--'
    )
    return precision_score(df3.label, df3.predicted), recall_score(df3.label, df3.predicted), f1_score(df3.label, df3.predicted), accuracy_score(df3.label, df3.predicted), coverage

def get_performance_metrics_BF_PPC_on(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    x_axis = []
    f1 = []
    accuracy = []
    coverage_ = []
    
    for i in range(1, 51):
        inc = i/100
        upper = 0.5 + inc
        lower = 0.5 - inc

        df1_condition = (df1['bayesfactor'].str.contains('inf') | df1['bayesfactor'].str.contains('NaNe-Inf'))
        df1_lower = df1['max_test'] >= lower
        df1_upper = df1['max_test'] <= upper
        df1_likelihood = df1['likelihood_e1_arrow_e2'] == 0
        df1_final = df1.loc[(df1_condition & df1_lower & df1_upper & df1_likelihood) | ~df1_likelihood]
        
        df2_condition = (df2['bayesfactor'].str.contains('inf') | df2['bayesfactor'].str.contains('NaNe-Inf'))
        df2_lower = df2['min_test'] >= lower
        df2_upper = df2['min_test'] <= upper
        df2_likelihood = df2['likelihood_e2_arrow_e1'] == 0
        df2_final = df2.loc[(df2_condition & df2_lower & df2_upper & df2_likelihood) | ~df2_likelihood]
        
        df3 = pd.concat([df1_final[["sentence_source", "label", "predicted"]], df2_final[["sentence_source", "label", "predicted"]]]).drop_duplicates(subset=['sentence_source'])
        coverage = len(df3.sentence_source)/len(df1.index)
        
        x_axis.append('50% \u00B1 ' + str(int(round(100 * inc, 0))) + "%")
        f1.append(f1_score(df3.label, df3.predicted))
        accuracy.append(accuracy_score(df3.label, df3.predicted))
        coverage_.append(coverage)

        
    return f1, accuracy, x_axis, coverage_

def get_performance_metrics_BF_on_PPC_on(file1, file2, i):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    upper = 0.5 + 0.5
    lower = 0.5 - 0.5

    df1_condition = (df1['bayesfactor'].str.contains('inf') | df1['bayesfactor'].str.contains('NaNe-Inf'))
    df1_lower = df1['max_test'] >= lower
    df1_upper = df1['max_test'] <= upper
    df1_likelihood = df1['likelihood_e1_arrow_e2'] == 0
    df1_final = df1.loc[(df1_condition & df1_lower & df1_upper & df1_likelihood) | ~df1_likelihood]
    
    df2_condition = (df2['bayesfactor'].str.contains('inf') | df2['bayesfactor'].str.contains('NaNe-Inf'))
    df2_lower = df2['min_test'] >= lower
    df2_upper = df2['min_test'] <= upper
    df2_likelihood = df2['likelihood_e2_arrow_e1'] == 0
    df2_final = df2.loc[(df2_condition & df2_lower & df2_upper & df2_likelihood) | ~df2_likelihood]
        
    df3 = pd.concat([df1_final[["sentence_source", "label", "predicted", "bayesfactor", "max_test", "min_test", "likelihood_e1_arrow_e2", "sentence"]],
                     df2_final[["sentence_source", "label", "predicted", "bayesfactor", "max_test", "min_test", "likelihood_e2_arrow_e1", "sentence"]]]).drop_duplicates(subset=['sentence_source'])
    #df3.to_csv('test.csv', index=False)
    cm = confusion_matrix(df3.label, df3.predicted)
    misclassify = cm[0,1] + cm[1,0]
    total = df1.shape[0]
    total_predicted = df3.shape[0]
    rejected = total - total_predicted
        
    coverage = len(df3.sentence_source)/len(df1.index)
    print(
        i, ",",
        '%.2f' % (100 * precision_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * recall_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * f1_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * accuracy_score(df3.label, df3.predicted)) + "\%", ",",
        '%.2f' % (100 * coverage) + "\%" + ",",
        '%.0f' % misclassify + " " + '%.0f' % rejected
    )
    return precision_score(df3.label, df3.predicted), recall_score(df3.label, df3.predicted), f1_score(df3.label, df3.predicted), accuracy_score(df3.label, df3.predicted), coverage

def test(file1, file2, i):
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.concat([df1[["sentence_source", "label", "predicted"]], df2[["sentence_source", "label", "predicted"]]]).drop_duplicates(subset=['sentence_source'])
    
    n_list = []
    for i in range(0, len(df3.label)):
        n = random.uniform(0, 1)
        if n > 0.5:
            n_list.append(1)
        else:
            n_list.append(0)

    return precision_score(df3.label, n_list), recall_score(df3.label, n_list), f1_score(df3.label, n_list), accuracy_score(df3.label, n_list), len(n_list)/len(df3.label)

if __name__ == "__main__":
    
    Data_Augmentation_Bayesian_inference_PPC = False
    Data_Augmentation_Bayesian_inference_BF_on = False
    Data_Augmentation_Bayesian_inference_BF_off = False
    Data_Augmentation_Bayesian_BERT = False
    Data_Augmentation_Bayesian_inference_BF_on_PPC_on = False
    Data_Augmentation_Bayesian_Loss = False
    
    Zero_Shot_Bayesian_inference_BF_on_PPC_on_plot = False
    Zero_Shot_Bayesian_inference_PPC = False
    Zero_Shot_Bayesian_inference_BF_on_PPC_on = False
    Zero_Shot_Bayesian_inference_PPC_results = False
    Zero_Shot_Bayesian_Loss = True
    
    Random = False
    
    if Data_Augmentation_Bayesian_inference_PPC:
        get_Model_Evaluation('/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Data Augmentation/Bayesian Inference (Unmodified Likelihoods)/e1_arrow_e2_Bayesian_10.csv',
                             '/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Data Augmentation/Bayesian Inference (Unmodified Likelihoods)/e2_arrow_e1_Bayesian_10.csv')
    
    if Data_Augmentation_Bayesian_inference_BF_on:
        precision_list, recall_list, f1_list, accuracy_list, coverage_list = [], [], [], [], []
        print("Run, Precision, Recall, F1, Accuracy, Coverage, Loss")
        for i in range(1,11):
            precision, recall, f1, accuracy, coverage = get_performance_metrics_BF_on('/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Data Augmentation/Bayesian Inference (Unmodified Likelihoods)/e1_arrow_e2_Bayesian_' + str(i) + '.csv',
                                                                                      '/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Data Augmentation/Bayesian Inference (Unmodified Likelihoods)/e2_arrow_e1_Bayesian_' + str(i) + '.csv',
                                                                                      i)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
            coverage_list.append(coverage)
        print(
            "Average", ",",
            '%.2f' % (100 * statistics.mean(precision_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(recall_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(f1_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(accuracy_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(coverage_list)) + "\%", ",",
            '--'
        )
        print(
            "SD", ",",
            "(" + '%.2f' % (100 * statistics.stdev(precision_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(recall_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(f1_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(accuracy_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(coverage_list)) + "\%)", ",",
            '--'
        )   
    
    if Data_Augmentation_Bayesian_inference_BF_off:
        precision_list, recall_list, f1_list, accuracy_list, coverage_list = [], [], [], [], []
        print("Run, Precision, Recall, F1, Accuracy, Coverage, Loss")
        for i in range(1,11):
            precision, recall, f1, accuracy, coverage = get_performance_metrics_BF_off('/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Data Augmentation/Bayesian Inference (Unmodified Likelihoods)/e1_arrow_e2_Bayesian_' + str(i) + '.csv',
                                                                                       '/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Data Augmentation/Bayesian Inference (Unmodified Likelihoods)/e2_arrow_e1_Bayesian_' + str(i) + '.csv',
                                                                                       i)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
            coverage_list.append(coverage)
        print(
            "Average", ",",
            '%.2f' % (100 * statistics.mean(precision_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(recall_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(f1_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(accuracy_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(coverage_list)) + "\%", ",",
            '--'
        )
        print(
            "SD", ",",
            "(" + '%.2f' % (100 * statistics.stdev(precision_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(recall_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(f1_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(accuracy_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(coverage_list)) + "\%)", ",",
            '--'
        )   

    if Data_Augmentation_Bayesian_BERT:
        f1, accuracy, coverage = [], [], []
        for i in range(1,11):
            f1_temp, accuracy_temp, x_axis, coverage_temp = get_performance_metrics_BF_PPC_on('/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Data Augmentation/Bayesian Inference (Unmodified Likelihoods)/e1_arrow_e2_Bayesian_' + str(i) + '.csv',
                                                                               '/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Data Augmentation/Bayesian Inference (Unmodified Likelihoods)/e2_arrow_e1_Bayesian_' + str(i) + '.csv')
            f1.append(f1_temp)
            accuracy.append(accuracy_temp)
            coverage.append(coverage_temp)

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(x_axis, coverage[0], 'red')
        ax2.plot(x_axis, f1[0], 'blue')
        
        ax1.set_xlabel('Range')
        ax1.set_ylabel('F1 Score', color='blue')
        ax2.set_ylabel('Coverage', color='red')
        
        plt.xticks([0, 9, 19, 29, 39, 49])
        plt.yticks(np.arange(0, 1, 0.1))
        #plt.ylabel('F1 Score', fontsize=12)
        #plt.xlabel('Range', fontsize=12)
        plt.show()
    
    if Data_Augmentation_Bayesian_inference_BF_on_PPC_on:
        precision_list, recall_list, f1_list, accuracy_list, coverage_list = [], [], [], [], []
        print("Run, Precision, Recall, F1, Accuracy, Coverage, Loss")
        for i in range(1,11):
            precision, recall, f1, accuracy, coverage = get_performance_metrics_BF_on_PPC_on('/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Data Augmentation/Bayesian Inference (Unmodified Likelihoods)/e1_arrow_e2_Bayesian_' + str(i) + '.csv',
                                                                                             '/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Data Augmentation/Bayesian Inference (Unmodified Likelihoods)/e2_arrow_e1_Bayesian_' + str(i) + '.csv',
                                                                                             i)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
            coverage_list.append(coverage)
        print(
            "Average", ",",
            '%.2f' % (100 * statistics.mean(precision_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(recall_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(f1_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(accuracy_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(coverage_list)) + "\%", ",",
            '--'
        )
        print(
            "SD", ",",
            "(" + '%.2f' % (100 * statistics.stdev(precision_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(recall_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(f1_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(accuracy_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(coverage_list)) + "\%)", ",",
            '--'
        )   

    if Data_Augmentation_Bayesian_Loss:
        lambda_e = list(range(0, 100))
        lambda_r = list(range(0, 100))

        loss1 = [ [0]*100 for i in range(100)]
        loss2 = [ [0]*100 for i in range(100)]
        loss3 = [ [0]*100 for i in range(100)]
        
        for e in lambda_e:
            for r in lambda_r:
                total = 2 * e + 293 * r
                loss1[e][r] = total

        for e in lambda_e:
            for r in lambda_r:
                total = 8 * e + 276 * r
                loss2[e][r] = total
                
        for e in lambda_e:
            for r in lambda_r:
                total = 113 * e + 43 * r
                loss3[e][r] = total

                
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)  
        ax.set_ylabel("\u03BBe")
        
        ax1 = fig.add_subplot(211)
        ax1 = sns.heatmap(loss1, vmin=0, vmax=30000, cmap =sns.cm.rocket_r)
        one_patch = mpatches.Patch(color = '#FF000000', label='2 * \u03BBe + 293 * \u03BBr')
        ax1.legend(handles=[one_patch])
        ax1.invert_yaxis()

        #ax2 = fig.add_subplot(212)
        #ax2 = sns.heatmap(loss2, vmin=0, vmax=30000, cmap =sns.cm.rocket_r)
        #two_patch = mpatches.Patch(color = '#FF000000', label='8 * \u03BBe + 276 * \u03BBr')
        #ax2.legend(handles=[two_patch])
        #ax2.invert_yaxis()

        ax3 = fig.add_subplot(212)
        ax3 = sns.heatmap(loss3, vmin=0, vmax=30000, cmap =sns.cm.rocket_r)
        three_patch = mpatches.Patch(color = '#FF000000', label='113 * \u03BBe + 43 * \u03BBr')
        ax3.legend(handles=[three_patch])
        ax3.set_xlabel('\u03BBr')
        ax3.invert_yaxis()
        
        plt.show()
        
    if Zero_Shot_Bayesian_inference_BF_on_PPC_on_plot:
        f1, accuracy, coverage = [], [], []
        for i in range(1,11):
            f1_temp, accuracy_temp, x_axis, coverage_temp = get_performance_metrics_BF_PPC_on('/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Zero-Shot Learning/Bayesian Inference/e1_arrow_e2_Bayesian_' + str(i) + '.csv',
                                                                                              '/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Zero-Shot Learning/Bayesian Inference/e2_arrow_e1_Bayesian_' + str(i) + '.csv')
            f1.append(f1_temp)
            accuracy.append(accuracy_temp)
            coverage.append(coverage_temp)

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(x_axis, coverage[0], 'red')
        ax2.plot(x_axis, f1[0], 'blue')
        
        ax1.set_xlabel('Range')
        ax1.set_ylabel('F1 Score', color='blue')
        ax2.set_ylabel('Coverage', color='red')
        
        plt.xticks([0, 9, 19, 29, 39, 49])
        plt.yticks(np.arange(0, 1, 0.1))
        #plt.ylabel('F1 Score', fontsize=12)
        #plt.xlabel('Range', fontsize=12)
        plt.show()

    if Zero_Shot_Bayesian_inference_PPC:
        precision_list, recall_list, f1_list, accuracy_list, coverage_list = [], [], [], [], []
        print("Run, Precision, Recall, F1, Accuracy, Coverage, Loss")
        for i in range(1,11):
            precision, recall, f1, accuracy, coverage = get_performance_metrics_BF_off('/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Zero-Shot Learning/Bayesian Inference/e1_arrow_e2_Bayesian_' + str(i) + '.csv',
                                                                                       '/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Zero-Shot Learning/Bayesian Inference/e2_arrow_e1_Bayesian_' + str(i) + '.csv',
                                                                                       i)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
            coverage_list.append(coverage)
        print(
            "Average", ",",
            '%.2f' % (100 * statistics.mean(precision_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(recall_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(f1_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(accuracy_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(coverage_list)) + "\%", ",",
            '--'
        )
        print(
            "SD", ",",
            "(" + '%.2f' % (100 * statistics.stdev(precision_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(recall_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(f1_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(accuracy_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(coverage_list)) + "\%)", ",",
            '--'
        )   

    if Zero_Shot_Bayesian_inference_BF_on_PPC_on:
        precision_list, recall_list, f1_list, accuracy_list, coverage_list = [], [], [], [], []
        print("Run, Precision, Recall, F1, Accuracy, Coverage, Loss")
        for i in range(1,11):
            precision, recall, f1, accuracy, coverage = get_performance_metrics_BF_on_PPC_on('/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Zero-Shot Learning/Bayesian Inference/e1_arrow_e2_Bayesian_' + str(i) + '.csv',
                                                 '/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Zero-Shot Learning/Bayesian Inference/e2_arrow_e1_Bayesian_' + str(i) + '.csv',
                                                 i)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
            coverage_list.append(coverage)
        print(
            "Average", ",",
            '%.2f' % (100 * statistics.mean(precision_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(recall_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(f1_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(accuracy_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(coverage_list)) + "\%", ",",
            '--'
        )
        print(
            "SD", ",",
            "(" + '%.2f' % (100 * statistics.stdev(precision_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(recall_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(f1_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(accuracy_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(coverage_list)) + "\%)", ",",
            '--'
        )

    if Zero_Shot_Bayesian_inference_PPC_results:
        get_Model_Evaluation('/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Zero-Shot Learning/Bayesian Inference/e1_arrow_e2_Bayesian_10.csv',
                             '/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Zero-Shot Learning/Bayesian Inference/e2_arrow_e1_Bayesian_10.csv')
            
    
    if Zero_Shot_Bayesian_Loss:
        lambda_e = list(range(0, 100))
        lambda_r = list(range(0, 100))

        loss1 = [ [0]*100 for i in range(100)]
        loss2 = [ [0]*100 for i in range(100)]
        loss3 = [ [0]*100 for i in range(100)]
        
        for e in lambda_e:
            for r in lambda_r:
                total = 2 * e + 321 * r
                loss1[e][r] = total

        for e in lambda_e:
            for r in lambda_r:
                total = 10 * e + 300 * r
                loss2[e][r] = total
                
        for e in lambda_e:
            for r in lambda_r:
                total = 125 * e + 48 * r
                loss3[e][r] = total

                
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)  
        ax.set_ylabel("\u03BBe")
        
        ax1 = fig.add_subplot(211)
        ax1 = sns.heatmap(loss1, vmin=0, vmax=30000, cmap =sns.cm.rocket_r)
        one_patch = mpatches.Patch(color = '#FF000000', label='2 * \u03BBe + 321 * \u03BBr')
        ax1.legend(handles=[one_patch])
        ax1.invert_yaxis()

        #ax2 = fig.add_subplot(212)
        #ax2 = sns.heatmap(loss2, vmin=0, vmax=30000, cmap =sns.cm.rocket_r)
        #two_patch = mpatches.Patch(color = '#FF000000', label='10 * \u03BBe + 300 * \u03BBr')
        #ax2.legend(handles=[two_patch])
        #ax2.invert_yaxis()

        ax3 = fig.add_subplot(212)
        ax3 = sns.heatmap(loss3, vmin=0, vmax=30000, cmap =sns.cm.rocket_r)
        three_patch = mpatches.Patch(color = '#FF000000', label='125 * \u03BBe + 48 * \u03BBr')
        ax3.legend(handles=[three_patch])
        ax3.set_xlabel('\u03BBr')
        ax3.invert_yaxis()
        
        plt.show()

    if Random:
        precision_list, recall_list, f1_list, accuracy_list, coverage_list = [], [], [], [], []
        i = 1
        print("Run, Precision, Recall, F1, Accuracy, Coverage, Loss")
        for j in range(0, 10000):
            precision, recall, f1, accuracy, coverage = test('/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Zero-Shot Learning/Bayesian Inference/e1_arrow_e2_Bayesian_' + str(i) + '.csv',
                                                             '/Users/jasonng/Documents/GitHub/Causal-Relation/data/SemEval2010_task8/result/Zero-Shot Learning/Bayesian Inference/e2_arrow_e1_Bayesian_' + str(i) + '.csv',
                                                             i)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
            coverage_list.append(coverage)
        
        print(
            "Average", ",",
            '%.2f' % (100 * statistics.mean(precision_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(recall_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(f1_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(accuracy_list)) + "\%", ",",
            '%.2f' % (100 * statistics.mean(coverage_list)) + "\%", ",",
            '--'
        )
        print(
            "SD", ",",
            "(" + '%.2f' % (100 * statistics.stdev(precision_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(recall_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(f1_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(accuracy_list)) + "\%)", ",",
            "(" + '%.2f' % (100 * statistics.stdev(coverage_list)) + "\%)", ",",
            '--'
        )
