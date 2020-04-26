import torch
import transformers as ppb


def make_bert_model(distil_bert=True):
    if distil_bert:
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-multilingual-cased')
    else:
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-multilingual-cased')

    model = model_class.from_pretrained(
        pretrained_weights,
        output_attentions=False,
        output_hidden_states=True,
    )
    model.eval()
    return model


def create_attention_mask_from(input_tensor):
    return torch.arange(input_tensor.size(1))[None, :] < input_tensor[:, None]


def get_bert_classification_vectors(hidden_states, use_cuda=False):
    if use_cuda:
        return hidden_states[0][:, 0, :].to('cpu').numpy()
    else:
        return hidden_states[0][:, 0, :].numpy()
