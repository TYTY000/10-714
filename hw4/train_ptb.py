import needle as ndl
import sys
sys.path.append('./apps')
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

device = ndl.cuda()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=16, device=ndl.cuda(), dtype="float32")
model = LanguageModel(30, len(corpus.dictionary), hidden_size=10, num_layers=2, seq_model='rnn', device=ndl.cuda())
train_ptb(model, train_data, seq_len=1, n_epochs=10, device=device)
evaluate_ptb(model, train_data, seq_len=40, device=device)
