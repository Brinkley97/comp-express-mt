import sklearn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class EvaluationMetric:

    def eval_accuracy(self, y_true, y_prediction):
        return accuracy_score(y_true, y_prediction)

    def eval_precision(self, y_true, y_prediction):
        return precision_score(y_true, y_prediction)

    def eval_recall(self, y_true, y_prediction):
        return recall_score(y_true, y_prediction)

    def eval_f1_score(self, y_true, y_prediction):
        return f1_score(y_true, y_prediction)
    
    def custom_evaluation_metrics(self, y_true, y_prediction):
        """Evaluate the model using accuracy and precision"""

        accuracy = self.eval_accuracy(y_true, y_prediction)
        precision = self.eval_precision(y_true, y_prediction)
        recall = self.eval_recall(y_true, y_prediction)
        f1 = self.eval_f1_score(y_true, y_prediction)

        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        return metrics_dict

    def eval_classification_report(self, y_true, y_prediction):
        print(classification_report(y_true, y_prediction))