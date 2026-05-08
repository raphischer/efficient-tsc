import os
import time
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from models.torch_utils import SaveBestModel, save_model

NEG_METRICS = {'loss'}  # metrics for which "better" is less



class Analyzer_Light(object):

    def __init__(self, maxcharlength=35, plot=False, output_filepath=None):

        self.maxcharlength = maxcharlength
        self.plot = plot
    
    def get_avg_prec_recall(self, ConfMatrix, existing_class_names, excluded_classes=None):
        """Get average recall and precision, using class frequencies as weights, optionally excluding
        specified classes"""

        class2ind = dict(zip(existing_class_names, range(len(existing_class_names))))
        included_c = np.full(len(existing_class_names), 1, dtype=bool)

        if not (excluded_classes is None):
            excl_ind = [class2ind[excl_class] for excl_class in excluded_classes]
            included_c[excl_ind] = False

        pred_per_class = np.sum(ConfMatrix, axis=0)
        nonzero_pred = (pred_per_class > 0)

        included = included_c & nonzero_pred
        support = np.sum(ConfMatrix, axis=1)
        weights = support[included] / np.sum(support[included])

        prec = np.diag(ConfMatrix[included, :][:, included]) / pred_per_class[included]
        prec_avg = np.dot(weights, prec)

        # rec = np.diag(ConfMatrix[included_c,:][:,included_c])/support[included_c]
        rec_avg = np.trace(ConfMatrix[included_c, :][:, included_c]) / np.sum(support[included_c])

        return prec_avg, rec_avg

    def analyze_classification(self, y_pred, y_true, class_names, excluded_classes=None):
        """
        For an array of label predictions and the respective true labels, shows confusion matrix, accuracy, recall, precision etc:
        Input:
            y_pred: 1D array of predicted labels (class indices)
            y_true: 1D array of true labels (class indices)
            class_names: 1D array or list of class names in the order of class indices.
                Could also be integers [0, 1, ..., num_classes-1].
            excluded_classes: list of classes to be excluded from average precision, recall calculation (e.g. OTHER)
        """

        # Trim class_names to include only classes existing in y_pred OR y_true
        in_pred_labels = set(list(y_pred))
        in_true_labels = set(list(y_true))

        self.existing_class_ind = sorted(list(in_pred_labels | in_true_labels))
        class_strings = [str(name) for name in class_names]  # needed in case `class_names` elements are not strings
        self.existing_class_names = [class_strings[ind][:min(self.maxcharlength, len(class_strings[ind]))] for ind in self.existing_class_ind]  # a little inefficient but inconsequential

        # Confusion matrix
        ConfMatrix = confusion_matrix(y_true, y_pred)

        # Analyze results
        self.total_accuracy = np.trace(ConfMatrix) / len(y_true)
        # print('Overall accuracy: {:.3f}\n'.format(self.total_accuracy))

        # returns metrics for each class, in the same order as existing_class_names
        self.precision, self.recall, self.f1, self.support = precision_recall_fscore_support(y_true, y_pred, labels=self.existing_class_ind, zero_division=0)

        # Calculate average precision and recall
        self.prec_avg, self.rec_avg = self.get_avg_prec_recall(ConfMatrix, self.existing_class_names, excluded_classes)
        if excluded_classes:
            print(
                "\nAverage PRECISION: {:.2f}\n(using class frequencies as weights, excluding classes with no predictions and predictions in '{}')".format(
                    self.prec_avg, ', '.join(excluded_classes)))
            print(
                "\nAverage RECALL (= ACCURACY): {:.2f}\n(using class frequencies as weights, excluding classes in '{}')".format(
                    self.rec_avg, ', '.join(excluded_classes)))

        # Make a histogram with the distribution of classes with respect to precision and recall
        #self.prec_rec_histogram(self.precision, self.recall)

        return {"total_accuracy": self.total_accuracy, "precision": self.precision, "recall": self.recall,
                "f1": self.f1, "support": self.support, "prec_avg": self.prec_avg, "rec_avg": self.rec_avg,
                "ConfMatrix": ConfMatrix}



def validate(val_evaluator, config, best_metrics, best_value, epoch, key_metric='loss'):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""
    with torch.no_grad():
        aggr_metrics, ConfMat = val_evaluator.evaluate(epoch, keep_all=False)
    print_str = 'Validation Summary: '
    for k, v in aggr_metrics.items():
        # tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)

    if key_metric in NEG_METRICS:
        condition = (aggr_metrics[key_metric] < best_value)
    else:
        condition = (aggr_metrics[key_metric] > best_value)
    if condition:
        best_value = aggr_metrics[key_metric]
        save_model(os.path.join(config['output_dir'], f'fold_{config["fold"]}', 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

    return aggr_metrics, best_metrics, best_value


class BaseRunner(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10, console=False):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.epoch_metrics = OrderedDict()



    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class SupervisedRunner(BaseRunner):

    def __init__(self, *args, **kwargs):

        super(SupervisedRunner, self).__init__(*args, **kwargs)

        self.analyzer = Analyzer_Light()

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):
            X, targets, _ = batch
            targets = targets.to(self.device)
            # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
            predictions = self.model(X.to(self.device))

            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            total_loss = batch_loss / len(loss)  # mean loss (over samples) used for optimization

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += 1
                epoch_loss += total_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=False, fsize=-1):
        self.model = self.model.eval()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):
            X, targets, _ = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            per_batch['targets'].append(targets.cpu().numpy())
            predictions = predictions.detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append([loss.cpu().numpy()])
            # per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)
        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes
        if keep_all:
            self.epoch_metrics['bal_acc'] = balanced_accuracy_score(targets, predictions)
            
            '''
            if len(np.unique(targets)) > 2:
                self.epoch_metrics['auc'] = roc_auc_score(targets, probs, multi_class="ovr")
                self.epoch_metrics['logloss'] = log_loss(targets, probs)
            else:
                self.epoch_metrics['auc'] = roc_auc_score(targets, predictions)
                self.epoch_metrics['logloss'] = log_loss(targets, predictions)
            '''

            self.epoch_metrics['weighted_f1'] = f1_score(targets, predictions, average="weighted")
            self.epoch_metrics['micro_f1'] = f1_score(targets, predictions, average="micro")
            self.epoch_metrics['macro_f1'] = f1_score(targets, predictions, average="macro")
            self.epoch_metrics['file_size'] = fsize
            metrics_dict['targets'] = targets
            metrics_dict['predictions'] = predictions
            metrics_dict['probs'] = probs
        return self.epoch_metrics, metrics_dict
    
def train_runner(config, model, trainer, val_evaluator, optimizer, path):
    epochs = config["n_epochs"]
    total_start_time = time.time()
    best_value = 1e16
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
    save_best_model = SaveBestModel()
    start_epoch = 0
    # save_best_acc_model = utils.SaveBestACCModel()

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, config, best_metrics,
                                                              best_value, epoch)
        save_best_model(aggr_metrics_val['loss'], epoch, model, optimizer, torch.nn.CrossEntropyLoss(), path)
        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))

    total_runtime = time.time() - total_start_time
    return model, total_runtime
