# Lab_1

Determine if product reviews are positive or negative.

`botnet.py`  trains a simple net to the assignment

`chatbot.py`  a simple chatbot reading a line, replies accordingly. Can use both simple and transformer models.

## Transformer

`transformer.py` maximum overkill, misuses a transformer to do the same thing

The decode side input is wired to an embedded input symbol `SOS`.
Thus only one output is produced, logits for two classes - positive or negative.

 ## Noteable Features used

### Lightning

* EarlyStopping
* ModelCheckpoint (best model selection)
* tensorboard logging
* tensorboard graph of model - transformer is interesting
