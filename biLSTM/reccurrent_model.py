import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm

#Funcion que crea un vocabulario de palabras con un indice numerico
def vocab():
    vocab = defaultdict()
    vocab.default_factory = lambda: len(vocab)
    return vocab    

#Funcion que pasa la cadena de simbolos a una secuencia con indices numericos
def text2numba(corpus, vocab):
    for doc in corpus:
        yield [vocab[w] for w in doc]
        
class biLSTM():
	def __init__(self, sents, tags, dim_emb=100, dim_lstm=200):
		super().__init__()
		#Crear indices de las cadenas
		self.input_voc = vocab()
		self.output_voc = vocab()
		self.in_idx = list(text2numba(sents, self.input_voc))
		self.out_idx = list(text2numba(tags, self.output_voc))
		self.ret_tags = {v:k for k,v in self.output_voc.items()}
		self.input_voc['OOV'] = len(self.input_voc)
		#Obtener train set
		self.train_pairs = list(zip(self.in_idx,self.out_idx))
		#Capa de embedding
		self.emb = nn.Sequential(nn.Embedding(len(self.input_voc), dim_emb))
		#Una capa de biLSTM
		self.bilstm = nn.LSTM(dim_emb, dim_lstm, bidirectional=True)
		#Salida con Softmax
		self.outLayer = nn.Sequential(nn.Linear(2*dim_lstm, len(self.output_voc)), nn.Softmax(dim=2))
		
	def train_model(self, its=100, lr=0.1):
		#Risk = Cross entropy
		criterion = nn.CrossEntropyLoss()
		#Optimizer = Stochastic Gradien Descent
		optimizer = torch.optim.SGD(list(self.emb.parameters()) + list(self.bilstm.parameters()) + list(self.outLayer.parameters()), lr=0.1)
		
		#Train the network
		for epoch in tqdm(range(its)):
			for sent, pred in self.train_pairs:
				#forward
				sent = torch.tensor(sent)
				emb_x = self.emb(sent)
				emb_x = emb_x.unsqueeze(1)
				out, dat = self.bilstm(emb_x)
				y_pred = self.outLayer(out)
				y_pred = y_pred.transpose(1,2)
				
				#backward
				pred = (torch.tensor(pred)).unsqueeze(1)
				loss = criterion(y_pred, pred)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
	
	#Prediction forward			
	def forward(self, string):
		sentOOV = []
		for w in string:
			if w in self.input_voc.keys():
				sentOOV.append(w)
			else:
				sentOOV.append('OOV')
		sent = torch.tensor(list(text2numba([sentOOV],self.input_voc))[0])
		emb_x = self.emb(sent)
		emb_x = emb_x.unsqueeze(1)
		out, (hn, cn) = self.bilstm(emb_x)
		y_pred = self.outLayer(out)
		y_pred = y_pred.transpose(1,2)
		
		tags = [self.ret_tags[int(y.argmax())] for y in y_pred]

		return tags
