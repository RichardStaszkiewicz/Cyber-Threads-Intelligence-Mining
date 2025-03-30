from collections import defaultdict
import numpy as np
import torch
import torchmetrics
from seqeval.metrics.sequence_labeling import get_entities


def get_entity_and_relation(label):
    """
    Extract entity type and relation type from label.
    
    Args:
        label (str): Entity label.
    
    Returns:
        tuple: (entity_type, relation_type)
    """
    parts = label.split('_')
    if label == 'O':  # Directly handle the 'O' (outside) tag
        return 'O', ''
    elif len(parts) == 2:
        return label, ''
    elif len(parts) > 2:
        return '_'.join(parts[:-2]), parts[-2]
    else:
        return None, None  # Handle invalid label formats


def extract_tp_actual_correct(y_true, y_pred, suffix=False):
    """
    Extract true positives, predicted counts, and actual counts.

    Args:
        y_true (list of lists): True entity labels.
        y_pred (list of lists): Predicted entity labels.
        suffix (bool): Whether labels end with '-B', '-I', '-E', '-S'.

    Returns:
        tuple: (pred_sum, tp_sum, true_sum)
    """
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)

    for sent_true, sent_pred in zip(y_true, y_pred):
        for label, start, end in get_entities(sent_true, suffix):
            entity_type, relation_type = get_entity_and_relation(label)
            entities_true[(entity_type, relation_type)].add((start, end))

        for label, start, end in get_entities(sent_pred, suffix):
            entity_type, relation_type = get_entity_and_relation(label)
            entities_pred[(entity_type, relation_type)].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
    
    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)

    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum


class ChunkEvaluator(torchmetrics.Metric):
    """
    ChunkEvaluator computes precision, recall, and F1-score for entity recognition tasks.
    
    Args:
        label_list (list): List of entity labels.
        suffix (bool): If True, labels end with '-B', '-I', '-E', '-S'. Otherwise, labels start with them.
        logger (wandb or any logger, optional): Logging object for tracking metrics.
    """

    def __init__(self, label_list, suffix=False, logger=None):
        super().__init__()
        self.id2label_dict = dict(enumerate(label_list))
        self.suffix = suffix
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0
        self.logger = logger  # Optional logger (e.g., wandb)

    def compute(self, lengths, predictions, labels):
        """
        Compute precision, recall, and F1-score for entity detection.

        Args:
            lengths (torch.Tensor): Valid sequence lengths, shape `[batch_size]`.
            predictions (torch.Tensor): Predicted indices, shape `[batch_size, sequence_length]`.
            labels (torch.Tensor): Ground truth labels, shape `[batch_size, sequence_length]`.

        Returns:
            tuple: (num_infer_chunks, num_label_chunks, num_correct_chunks)
        """
        labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()

        unpad_labels = [
            [self.id2label_dict[index] for index in labels[i][:lengths[i]]]
            for i in range(len(lengths))
        ]
        unpad_predictions = [
            [self.id2label_dict.get(index, "O") for index in predictions[i][:lengths[i]]]
            for i in range(len(lengths))
        ]

        pred_sum, tp_sum, true_sum = extract_tp_actual_correct(
            unpad_labels, unpad_predictions, self.suffix
        )

        num_correct_chunks = torch.tensor([tp_sum.sum()])
        num_infer_chunks = torch.tensor([pred_sum.sum()])
        num_label_chunks = torch.tensor([true_sum.sum()])

        return num_infer_chunks, num_label_chunks, num_correct_chunks

    def update(self, num_infer_chunks, num_label_chunks, num_correct_chunks):
        """
        Update chunk counts for precision, recall, and F1-score calculations.

        Args:
            num_infer_chunks (int or torch.Tensor): Number of predicted chunks.
            num_label_chunks (int or torch.Tensor): Number of true chunks.
            num_correct_chunks (int or torch.Tensor): Number of correctly predicted chunks.
        """
        self.num_infer_chunks += num_infer_chunks.item()
        self.num_label_chunks += num_label_chunks.item()
        self.num_correct_chunks += num_correct_chunks.item()

    def accumulate(self):
        """
        Compute overall precision, recall, and F1-score.

        Returns:
            tuple: (precision, recall, f1_score)
        """
        precision = (self.num_correct_chunks / self.num_infer_chunks) if self.num_infer_chunks else 0.0
        recall = (self.num_correct_chunks / self.num_label_chunks) if self.num_label_chunks else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if self.num_correct_chunks else 0.0

        # Log metrics if logger is provided
        if self.logger:
            self.logger.log({
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            })

        return precision, recall, f1_score

    def reset(self):
        """
        Reset evaluation metrics.
        """
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def name(self):
        """
        Return metric names.
        """
        return "precision", "recall", "f1"
