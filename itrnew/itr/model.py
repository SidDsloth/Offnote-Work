import torch
import pytorch_lightning as pl
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer
from data import IndicDataset
from easydict import EasyDict as ED
from torch.utils.data import DataLoader
class TranslationModel(pl.LightningModule):

    def __init__(self, encoder, decoder,config,tokenizers):

        super().__init__() 

        #Creating encoder and decoder with their respective embeddings.
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.tokenizers = tokenizers

    def forward(self, encoder_input_ids, decoder_input_ids):

        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states, 
                                    masked_lm_labels=decoder_input_ids)

        return loss, logits

    def save_model(model, output_dir):
      output_dir = Path(output_dir)
    # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
      model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
      output_model_file = output_dir / WEIGHTS_NAME
      output_config_file = output_dir / CONFIG_NAME

      torch.save(model_to_save.state_dict(), output_model_file)
      model_to_save.config.to_json_file(output_config_file)
    #src_tokenizer.save_vocabulary(output_dir)

    def save(self, tokenizers, output_dirs):
      from train_util import save_model
      save_model(self.encoder, output_dirs.encoder)
      save_model(self.decoder, output_dirs.decoder)
   
    def training_step(self,batch):
      source,target=batch
      loss, logits = self.forward(source, target)
      total_loss += loss.item()
      logits = logits.detach().cpu().numpy()
      label_ids = target.to('cpu').numpy()
      return loss
    def validation_step(self,batch):
      source,target=batch
      loss, logits = self.forward(source, target)
      total_loss += loss.item()
      logits = logits.detach().cpu().numpy()
      label_ids = target.to('cpu').numpy()
      return loss
    def validation_epoch_end(self, outputs):
      #avg_valid_acc = eval_accuracy/nb_eval_steps
      #validation_accuracy_values.append(avg_valid_acc)
      avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
      tensorboard_logs = {'val_loss': avg_loss}
      return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
      """writer.add_scalar('Valid/Accuracy', avg_valid_acc, epoch)
      writer.flush()
      print("Avg Val Accuracy: {0:.2f}".format(avg_valid_acc))
      print("Time taken by epoch: {0:.2f}".format(time.time() - start_time))
      """
    @pl.data_loader
    def train_dataloader(self):
      return DataLoader(IndicDataset(self.tokenizers.src,self.tokenizers.tgt,self.config.data, True), 
                            batch_size=self.config.batch_size, 
                            shuffle=False, 
                            collate_fn=pad_sequence)

    @pl.data_loader
    def val_dataloader(self):
      return DataLoader(IndicDataset(self.tokenizers.src,self.tokenizers.tgt,self.config.data, False), 
                           batch_size=self.config.eval_size, 
                           shuffle=False, 
                           collate_fn=pad_sequence)
     def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader()), eta_min=self.config.lr)
      return optimizer, scheduler

def build_model(config):
    
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tgt_tokenizer.bos_token = '<s>'
    tgt_tokenizer.eos_token = '</s>'
    
    #hidden_size and intermediate_size are both wrt all the attention heads. 
    #Should be divisible by num_attention_heads
    encoder_config = BertConfig(vocab_size=src_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12)

    decoder_config = BertConfig(vocab_size=tgt_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12,
                                is_decoder=True)

    #Create encoder and decoder embedding layers.
    encoder_embeddings = torch.nn.Embedding(src_tokenizer.vocab_size, config.hidden_size, padding_idx=src_tokenizer.pad_token_id)
    decoder_embeddings = torch.nn.Embedding(tgt_tokenizer.vocab_size, config.hidden_size, padding_idx=tgt_tokenizer.pad_token_id)

    encoder = BertModel(encoder_config)
    encoder.set_input_embeddings(encoder_embeddings.cuda())
    
    decoder = BertForMaskedLM(decoder_config)
    decoder.set_input_embeddings(decoder_embeddings.cuda())
    
    tokenizers = ED({'src': src_tokenizer, 'tgt': tgt_tokenizer})
    
    model = TranslationModel(encoder, decoder,config,tokenizers)
    model.cuda()
    
    return model, tokenizers

