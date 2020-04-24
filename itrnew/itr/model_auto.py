import torch
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer
from transformers import DistilBertConfig
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForMaskedLM
from transformers import GPT2Model, GPT2Config

from easydict import EasyDict as ED


def build_model_pretrained(config):

    #Create different tokenizers for both source and target language.
    src_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tgt_tokenizer.bos_token = '<s>'
    tgt_tokenizer.eos_token = '</s>'

    #encoder_config = DistilBertConfig.from_pretrained('distilbert-base-multilingual-cased')

    encoder = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

    if config.decoder.pretrained:
        decoder = BertForMaskedLM.from_pretrained('bert-base-uncased')
    else:

        decoder_config = BertConfig(vocab_size=tgt_tokenizer.vocab_size,
                              is_decoder=True)
        decoder = BertForMaskedLM(decoder_config)

    model = TranslationModel(encoder, decoder) 
    model.cuda()

    tokenizers = ED({'src': src_tokenizer, 'tgt': tgt_tokenizer})
    return model, tokenizers
