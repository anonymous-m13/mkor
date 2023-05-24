from .transformer import TransformerModel
from transformers import BertForPreTraining, BertForMaskedLM, BertConfig
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import models.bert.bert_nvidia as bert_nvidia
from transformers.modeling_outputs import MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
import torch


def bert_large_cased(**kwargs):
    """BERT-Large-Cased model."""
    config = AutoConfig.from_pretrained('bert-large-cased')
    model = BertForPreTraining(config)
    return model


def bert_base_cased(**kwargs):
    """BERT-Base-Cased model."""
    config = BertConfig.from_pretrained('bert-base-cased')
    model = BertForPreTraining(config)
    return model


def bert_large_uncased(**kwargs):
    """BERT-Large-Uncased model."""
    config = AutoConfig.from_pretrained('bert-large-uncased')
    model = BertForPreTraining(config)
    return model


def bert_base_uncased(**kwargs):
    """BERT-Base-Uncased model."""
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = BertForPreTraining(config)
    return model


def bert_base_cased_mlm(**kwargs):
    """BERT-Base-Uncased model."""
    config = BertConfig.from_pretrained('bert-base-cased')
    model = BertForMaskedLM(config)
    return model


def bert_large_cased_mlm(**kwargs):
    """BERT-Large-Uncased model."""
    config = BertConfig.from_pretrained('bert-large-cased')
    model = BertForMaskedLM(config)
    return model


def bert_base_uncased_mlm(**kwargs):
    """BERT-Base-Uncased model."""
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM(config)
    return model


def bert_large_uncased_mlm(**kwargs):
    """BERT-Large-Uncased model."""
    config = BertConfig.from_pretrained('bert-large-uncased')
    model = BertForMaskedLM(config)
    return model


def bert_base_cased_sequence_classification(**kwargs):
    """BERT-Base-Uncased model."""
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=kwargs['num_classes'])
    return model


def bert_large_cased_sequence_classification(**kwargs):
    """BERT-Large-Uncased model."""
    model = AutoModelForSequenceClassification.from_pretrained('bert-large-cased', num_labels=kwargs['num_classes'])
    return model


def bert_base_cased_question_answering(**kwargs):
    """BERT-Base-Uncased model."""
    model = AutoModelForQuestionAnswering.from_pretrained('bert-base-cased')
    return model


def bert_large_cased_question_answering(**kwargs):
    """BERT-Large-Uncased model."""
    model = AutoModelForQuestionAnswering.from_pretrained('bert-large-cased')
    return model


def transformer(**kwargs):
    kwargs['ntoken'] = len(kwargs['vocab'])
    kwargs['d_model'] = 256
    kwargs['nhead'] = 4
    kwargs['d_hid'] = 256
    kwargs['nlayers'] = 1
    kwargs['dropout'] = 0.2
    return TransformerModel(**kwargs)


class BERT_Pretraining_Interface(torch.nn.Module):
    def __init__(self, model, criterion):
        super(BERT_Pretraining_Interface, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, **inputs):
        outputs_dict = {}
        outputs = MaskedLMOutput()
        outputs_dict['labels'] = inputs['labels']
        outputs_dict['next_sentence_labels'] = inputs['next_sentence_labels']
        outputs_dict['logits'], outputs_dict['seq_relationship_score'] = self.model(**inputs)
        outputs.logits = outputs_dict['logits']
        outputs.loss = self.criterion(**outputs_dict)
        return outputs


class BERT_Question_Answering_Interface(torch.nn.Module):
    def __init__(self, model, criterion):
        super(BERT_Question_Answering_Interface, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, **inputs):
        outputs_dict = {}
        outputs = QuestionAnsweringModelOutput()
        outputs_dict['start_positions'] = inputs['start_positions']
        outputs_dict['end_positions'] = inputs['end_positions']
        outputs_dict['start_logits'], outputs_dict['end_logits'] = self.model(**inputs)

        outputs.start_logits, outputs.end_logits = outputs_dict['start_logits'], outputs_dict['end_logits']
        outputs.loss = self.criterion(**outputs_dict)
        return outputs


class BERT_Classification_Interface(torch.nn.Module):
    def __init__(self, model, criterion):
        super(BERT_Classification_Interface, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, **inputs):
        outputs_dict = {}
        outputs = SequenceClassifierOutput()
        outputs_dict['labels'] = inputs['labels']
        outputs_dict['logits'] = self.model(**inputs)
        outputs.logits = outputs_dict['logits']
        outputs.loss = self.criterion(**outputs_dict)
        return outputs


def bert_large_uncased_nvidia(**kwargs):
    """BERT-Large-Uncased model."""
    config = bert_nvidia.BertConfig.from_json_file("models/bert/config/bert_large_uncased_config.json")
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    bert_nvidia.ACT2FN["bias_gelu"] = bert_nvidia.bias_gelu_training
    model = bert_nvidia.BertForPreTraining(config)
    criterion = bert_nvidia.BertPretrainingCriterion(config.vocab_size)
    return BERT_Pretraining_Interface(model, criterion)


def bert_large_uncased_question_answering_nvidia(**kwargs):
    """BERT-Large-Uncased model."""
    config = bert_nvidia.BertConfig.from_json_file("models/bert/config/bert_large_uncased_config.json")
    pretraining_model = bert_nvidia.BertForMaskedLM(config)
    checkpoint = torch.load("data/bert_large_uncased_nvidia.pt")['model']
    checkpoint["bert.embeddings.word_embeddings.weight"] = checkpoint["bert.embeddings.word_embeddings.weight"][:config.vocab_size, :]
    checkpoint["cls.predictions.decoder.weight"] = checkpoint["cls.predictions.decoder.weight"][:config.vocab_size, :]
    checkpoint["cls.predictions.bias"] = checkpoint["cls.predictions.bias"][:config.vocab_size]
    config.next_sentence = False
    bert_nvidia.ACT2FN["bias_gelu"] = bert_nvidia.bias_gelu_training
    model = bert_nvidia.BertForQuestionAnswering(config)
    criterion = bert_nvidia.BertForQuestionAnsweringCriterion()
    return BERT_Question_Answering_Interface(model, criterion)


def bert_large_uncased_classification_nvidia(**kwargs):
    """BERT-Large-Uncased model."""
    config = bert_nvidia.BertConfig.from_json_file("models/bert/config/bert_large_uncased_config.json")
    bert_nvidia.ACT2FN["bias_gelu"] = bert_nvidia.bias_gelu_training
    pretraining_model = bert_nvidia.BertForMaskedLM(config)
    checkpoint = torch.load("data/bert_large_uncased_nvidia.pt")['model']
    checkpoint["bert.embeddings.word_embeddings.weight"] = checkpoint["bert.embeddings.word_embeddings.weight"][:config.vocab_size, :]
    checkpoint["cls.predictions.decoder.weight"] = checkpoint["cls.predictions.decoder.weight"][:config.vocab_size, :]
    checkpoint["cls.predictions.bias"] = checkpoint["cls.predictions.bias"][:config.vocab_size]
    pretraining_model.load_state_dict(checkpoint)

    model = bert_nvidia.BertForSequenceClassification(config, num_labels=kwargs['num_classes'])
    model.bert = pretraining_model.bert
    del pretraining_model
    criterion = bert_nvidia.BertForSequenceClassificationCriterion()
    return BERT_Classification_Interface(model, criterion)