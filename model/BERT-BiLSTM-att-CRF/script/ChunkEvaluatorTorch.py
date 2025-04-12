from collections import defaultdict
import numpy as np
import torch
from seqeval.metrics.sequence_labeling import get_entities

# Optional logger support (e.g., wandb)
def get_logger():
    import logging
    return logging.getLogger(__name__)

def get_entity_and_relation(label):
    """
    Extract entity and relation types from a label string.
    """
    if label == 'O':
        return 'O', ''
    parts = label.split('_')
    if len(parts) == 2:
        return label, ''
    elif len(parts) > 2:
        return '_'.join(parts[:-2]), parts[-2]
    else:
        return None, None

def extract_tp_actual_correct(y_true, y_pred, suffix=False):
    """
    Extracts true positives, predictions, and actual entity spans.
    Returns counts for computing precision, recall, and F1.
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
        true_set = entities_true.get(type_name, set())
        pred_set = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(true_set & pred_set))
        pred_sum = np.append(pred_sum, len(pred_set))
        true_sum = np.append(true_sum, len(true_set))

    return pred_sum, tp_sum, true_sum

class ChunkEvaluator:
    """
    ChunkEvaluator computes precision, recall, and F1 for sequence labeling (e.g., NER).
    Compatible with PyTorch workflows.
    """

    def __init__(self, label_list, suffix=False, logger=None):
        """
        Args:
            label_list (List[str]): List of label names.
            suffix (bool): Whether the label uses suffix format (e.g. "PER-B").
            logger (Logger, optional): Logger object for warnings.
        """
        self.id2label_dict = dict(enumerate(label_list))
        self.suffix = suffix
        self.logger = logger or get_logger()
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0
        self._has_warned = False

    def compute(self, lengths, predictions, labels, dummy=None):
        """
        Compute precision, recall, and F1 for a batch.

        Args:
            lengths (Tensor): [batch_size] valid sequence lengths.
            predictions (Tensor): [batch_size, seq_len] predicted label indices.
            labels (Tensor): [batch_size, seq_len] ground-truth label indices.
            dummy (Tensor, optional): Compatibility arg (ignored).
        """
        if dummy is not None:
            # Handle old-style call for backward compatibility
            dummy, lengths, predictions, labels = lengths, predictions, labels, dummy
            if not self._has_warned:
                self.logger.warning(
                    "Compatibility Warning: `ChunkEvaluator.compute` parameter order has changed. "
                    "Old order: (inputs, lengths, predictions, labels). New order: (lengths, predictions, labels)."
                )
                self._has_warned = True

        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(lengths):
            lengths = lengths.cpu().numpy()

        unpad_labels = [
            [self.id2label_dict[idx] for idx in labels[i][:lengths[i]]]
            for i in range(len(lengths))
        ]
        unpad_predictions = [
            [self.id2label_dict.get(idx, "O") for idx in predictions[i][:lengths[i]]]
            for i in range(len(lengths))
        ]

        pred_sum, tp_sum, true_sum = extract_tp_actual_correct(unpad_labels, unpad_predictions, self.suffix)

        num_correct_chunks = torch.tensor([tp_sum.sum()])
        num_infer_chunks = torch.tensor([pred_sum.sum()])
        num_label_chunks = torch.tensor([true_sum.sum()])

        return num_infer_chunks, num_label_chunks, num_correct_chunks

    def _is_number_or_matrix(self, var):
        def _is_number(x):
            return isinstance(x, (int, float, np.integer, np.floating)) or (
                isinstance(x, np.ndarray) and x.shape == (1,)
            )
        return _is_number(var) or isinstance(var, np.ndarray)

    def update(self, num_infer_chunks, num_label_chunks, num_correct_chunks):
        """
        Update cumulative counts for precision/recall/F1.
        """
        if not self._is_number_or_matrix(num_infer_chunks):
            raise ValueError("'num_infer_chunks' must be int or np.ndarray")
        if not self._is_number_or_matrix(num_label_chunks):
            raise ValueError("'num_label_chunks' must be int or np.ndarray")
        if not self._is_number_or_matrix(num_correct_chunks):
            raise ValueError("'num_correct_chunks' must be int or np.ndarray")

        self.num_infer_chunks += num_infer_chunks
        self.num_label_chunks += num_label_chunks
        self.num_correct_chunks += num_correct_chunks

    def accumulate(self):
        """
        Calculate overall precision, recall, and F1 score from accumulated stats.
        """
        precision = float(self.num_correct_chunks / self.num_infer_chunks) if self.num_infer_chunks else 0.0
        recall = float(self.num_correct_chunks / self.num_label_chunks) if self.num_label_chunks else 0.0
        f1 = 2 * precision * recall / (precision + recall) if self.num_correct_chunks else 0.0
        return precision, recall, f1

    def reset(self):
        """
        Reset all accumulated counts.
        """
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def name(self):
        return "precision", "recall", "f1"
