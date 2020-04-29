import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from data import IndicDataset, PadSequence
from config import replace, preEnc, preEncDec
from data import IndicDataset, PadSequence
import model as M

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

    def prepare_data(self):
        self.pad_sequence = PadSequence(self.tokenizers.src.pad_token_id,self.tokenizers.tgt.pad_token_id)

    def train_dataloader(self):
      return DataLoader(IndicDataset(self.tokenizers.src,self.tokenizers.tgt,self.config.data, True), 
                            batch_size=self.config.batch_size, 
                            shuffle=False, 
                            collate_fn=self.pad_sequence)

    def val_dataloader(self):
      return DataLoader(IndicDataset(self.tokenizers.src,self.tokenizers.tgt, self.config.data, False), 
                           batch_size=self.config.eval_size, 
                           shuffle=False, 
                           collate_fn=self.pad_sequence)


    def training_step(self,batch,batch_idx):
      source,target=batch
      loss, logits = self.forward(source, target)
      logs={'train_loss':loss}
      return {'loss':loss,'log':logs}

    def validation_step(self,batch,batch_idx):
      source,target=batch
      loss, logits = self.forward(source, target)
      return {'loss':loss}

    def validation_end(self,outputs):
      avg_loss=torch.stack([x['loss'] for x in outputs]).mean()
      tensorboard_logs={'val_loss':avg_loss}
      return {'avg_val_loss':avg_loss,'log':tensorboard_logs}

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
      return optimizer

def main():
    rconf = preEncDec
    encoder,decoder,tokenizers=M.build_model(rconf)
    model=TranslationModel(encoder,decoder,rconf,tokenizers)
    tb_logger=TensorBoardLogger('tb_logs',name='Bert_model')
    trainer=pl.Trainer(min_epochs=1,max_epochs=6, check_val_every_n_epoch=1, logger=tb_logger)
    trainer.fit(model)

if __name__ == '__main__':
    main()
