import torch
import torch.nn.functional as F
from transformers import (BertForSequenceClassification, BertTokenizer, BertConfig, 
						  BertPreTrainedModel, BertModel, AdamW, get_linear_schedule_with_warmup)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset, Sampler)
from torch import nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import Optional
import random, os, csv, logging, json, copy, math, operator
from queue import PriorityQueue
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.autograd import Variable

from tqdm.auto import tqdm, trange

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt = '%m/%d/%Y %H:%M:%S',
						level = logging.INFO)
logger = logging.getLogger(__name__)

global outputs_map
global output_vocab
global all_outputs
global num_ptrs
global num_slots
global num_intents



# Decoder from 'Attention is all you need'

def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return x + self.dropout(sublayer(self.norm(x)))

class Decoder(nn.Module):
	"Generic N layer decoder with masking."
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
		
	def forward(self, x, memory, src_mask, tgt_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)
	
class LayerNorm(nn.Module):
	"Construct a layernorm module (See citation for details)."
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
	
class DecoderLayer(nn.Module):
	"Decoder is made of self-attn, src-attn, and feed forward (defined below)"
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
	def forward(self, x, memory, src_mask, tgt_mask):
		"Follow Figure 1 (right) for connections."
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)
	
def subsequent_mask(size, batch_size=1):
	"Mask out subsequent positions."
	attn_shape = (batch_size, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)
		
		# 1) Do all the linear projections in batch from d_model => h x d_k 
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))]
		
		# 2) Apply attention on all the projected vectors in batch. 
		x, self.attn = attention(query, key, value, mask=mask, 
								 dropout=self.dropout)
		
		# 3) "Concat" using a view and apply a final linear. 
		x = x.transpose(1, 2).contiguous() \
			 .view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		return self.lut(x) * math.sqrt(self.d_model)
	
class PositionalEncoding(nn.Module):
	"Implement the PE function."
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) *
							 -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
		
	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)], 
						 requires_grad=False)
		return self.dropout(x)

class Generator(nn.Module):
	"Define standard linear + softmax generation step."
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
#         return F.log_softmax(self.proj(x), dim=-1)
		return self.proj(x)

class SimpleLossCompute:
	"A simple loss compute and train function."
	def __init__(self, generator, criterion, opt=None):
		self.generator = generator
		self.criterion = criterion
		self.opt = opt
		
	def __call__(self, x, y, norm):
		norm = norm.data.sum()
		loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
							  y.contiguous().view(-1)) / norm
		return loss

class LabelSmoothing(nn.Module):
	"Implement label smoothing."
	def __init__(self, size, padding_idx, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.criterion = nn.KLDivLoss(size_average=False)
		self.padding_idx = padding_idx
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.size = size
		self.true_dist = None
		
	def forward(self, x, target):
		assert x.size(1) == self.size
		true_dist = x.data.clone()
		true_dist.fill_(self.smoothing / (self.size - 3))
		true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		true_dist[:, self.padding_idx] = 0
		mask = torch.nonzero(target.data == self.padding_idx)
		if mask.dim() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		self.true_dist = true_dist
		res = self.criterion(x, Variable(true_dist, requires_grad=False))
		return res

def convert_output_to_ids(output):
	global outputs_map
	output_ids = []
	for w in output:
		output_ids.append(outputs_map[w])
	return output_ids

class BeamSearchNode(object):
	def __init__(self, ys, previousNode, wordId, logProb, length):
		'''
		:param hiddenstate:
		:param previousNode:
		:param wordId:
		:param logProb:
		:param length:
		'''
		self.ys = ys
		self.prevNode = previousNode
		self.wordid = wordId
		self.logp = logProb
		self.leng = length

	def eval(self, alpha=1.0):
		reward = 0
		# Add here a function for shaping a reward

		return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
	
	def __lt__(self, other):
		return True


# def beam_decode(target_tensor, decoder_hiddens, encoder_outputs=None):
def beam_decode(input_id, decoder_embed, decoder, encoder_outputs, encoder_outputs_excluding_sptokens, source_mask, generator):

	beam_width = 5
	topk = 1  # how many sentence do you want to generate
	decoded_batch = []
	start_symbol = 1
	end_symbol = 2
	batch_size = 1
	global num_ptrs
	num_ptrs = num_ptrs
	
	# decoding goes sentence by sentence
	for idx in range(encoder_outputs.size(0)):

		encoder_output = encoder_outputs[idx, : , :].unsqueeze(0) # [1, 128, 768]
		encoder_output_excluding_sptokens = encoder_outputs_excluding_sptokens[idx, : , :].unsqueeze(0)
		src_mask = source_mask[idx, :, :]

		# Number of sentence to generate
		endnodes = []
		number_required = min((topk + 1), topk - len(endnodes))

		# starting node -  hidden vector, previous node, word id, logp, length
		ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(input_id.data)
		
		node = BeamSearchNode(ys, None, start_symbol, 0, 1)
		nodes = PriorityQueue()

		# start the queue
		nodes.put((-node.eval(), node))
		qsize = 1
		
		tmpnodes = PriorityQueue()
		breaknow = False
		
		while not breaknow:
			
			nextnodes = PriorityQueue()
			
			# start beam search
			while nodes.qsize()>0:

				# fetch the best node
				score, n = nodes.get()
				prev_id = n.wordid
				ys = n.ys

				if (ys.shape[1]>100):
					breaknow = True
					break

				if n.wordid == end_symbol and n.prevNode != None:
					endnodes.append((score, n))
					# if we reached maximum # of sentences required
					if len(endnodes) == beam_width:
						breaknow = True
						break
					else:
						continue

				# decode for one step using decoder
				decoder_output = decoder(decoder_embed(Variable(ys)), encoder_output, src_mask,\
									   Variable(subsequent_mask(ys.size(1), batch_size=batch_size).type_as(input_id.data)))


				generator_scores = generator(decoder_output[:, -1])
				src_ptr_scores = torch.einsum('ac, adc -> ad', decoder_output[:, -1], encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])

				# Make src_ptr_scores of proper shape by appending 0's for non-existent pointers
				a, b = src_ptr_scores.shape
				src_ptr_scores_net = torch.zeros(a, num_ptrs).cuda()
				src_ptr_scores_net[:,:b] = src_ptr_scores

				all_scores = torch.cat((generator_scores, src_ptr_scores), axis=1)

				all_prob = F.log_softmax(all_scores, dim=-1)

				# PUT HERE REAL BEAM SEARCH OF TOP
				top_log_prob, top_indexes = torch.topk(all_prob, beam_width)

				for new_k in range(beam_width):
					decoded_t = top_indexes[0][new_k].view(1, -1).type_as(ys)
					log_prob = top_log_prob[0][new_k].item()

					ys2 = torch.cat([ys, decoded_t], dim=1)

					node = BeamSearchNode(ys2, n, decoded_t.item(), n.logp + log_prob, n.leng + 1)
					score = -node.eval()
					nextnodes.put((score, node))


				# put them into queue
				for i in range(beam_width):
					if nextnodes.qsize()>0:
						score, nn = nextnodes.get()
						nodes.put((score, nn))

		# choose nbest paths, back trace them
		if len(endnodes) == 0:
			endnodes = [nodes.get() for _ in range(topk)]
		utterances = []
		for score, n in sorted(endnodes, key=operator.itemgetter(0)):
			utterance = []
			utterance.append(n.wordid)
			# back trace
			while n.prevNode != None:
				n = n.prevNode
				utterance.append(n.wordid)

			utterance = utterance[::-1]
			utterances.append(utterance)
			if (len(utterances)==topk):
				break

		decoded_batch.append(utterances)

	return decoded_batch


class BertForSequenceGenerationWithPointerNet(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels # Includes both maxPtr# + intent labels + slot labels

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.tokenClassifier = nn.Linear(config.hidden_size, self.num_labels)
		
		# Decoder #
		N = 6 # stack of N=6 identical layers
		# The dimensionality of input and output is dmodel=512 , and the inner-layer has dimensionality dff=2048
		d_model = 768
		d_ff = 2048
		h=8 # parallel attention layers, or heads
		dropout=0.1
		global num_ptrs, all_outputs, output_vocab
		net_vocab = len(all_outputs)
		tgt_vocab = len(output_vocab)
		self.num_ptrs = num_ptrs
		c = copy.deepcopy
		attn = MultiHeadedAttention(h, d_model)
		ff = PositionwiseFeedForward(d_model, d_ff, dropout)
		position = PositionalEncoding(d_model, dropout)
		self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), 
							 c(ff), dropout), N=6)
		self.decoder_embed = nn.Sequential(Embeddings(d_model, net_vocab), c(position))
		
		
		# Loss
		criterion = LabelSmoothing(size=net_vocab, padding_idx=0, smoothing=0.1)
		self.generator = Generator(d_model, tgt_vocab)
		self.loss_compute = SimpleLossCompute(self.generator, criterion, None)
		###########

		self.init_weights()
		
	def getInputLogits(self, hidden, token_output):
		logits = []
		for output in token_output:
			logits.append(np.dot(hidden, output))
		return logits

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		source_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		output_ids=None, 
		output_attention_mask=None,
		output_ids_y=None, 
		output_mask=None,
		ntokens=None,
		input_length=None,
		output_length=None,
		decode=False,
		max_len=60, # 60 sufficient for valid
		start_symbol=1
	):
		
		szs = []
		
		max_inp_len = torch.max(input_length).cpu().item()
		max_out_len = torch.max(output_length).cpu().item()
		
		input_ids = input_ids[:, :max_inp_len]
		attention_mask = attention_mask[:, :max_inp_len]
		token_type_ids = token_type_ids[:, :max_inp_len]
		source_mask = source_mask[:,:,:max_inp_len]
		
		output_ids = output_ids[:, :max_out_len]
		output_ids_y = output_ids_y[:, :max_out_len]
		output_mask = output_mask[:, :max_out_len, :max_out_len]
		
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		
		encoder_output = outputs[0]
		decoder_output = self.decoder(self.decoder_embed(output_ids), encoder_output, source_mask,\
									  output_mask)
		
		tgt_vocab_scores = self.generator(decoder_output)
		norm = ntokens.data.sum()
		
		# Remove output for <start> and <end> tokens from encoder_output for excluding their pointer scores
		encoder_output_excluding_sptokens = encoder_output.clone()
		for i in range(encoder_output.shape[0]):
			encoder_output_excluding_sptokens[i,input_length[i]-1:,:] = 0.0 # removing output corresponding to <end> tokens for different seq lengths       
		encoder_output_excluding_sptokens = encoder_output_excluding_sptokens[:,1:-1,:] # removing output corresponding to <start> & <end> tokens as they all are aligned
		
		# Get pointer scores
		src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output, encoder_output_excluding_sptokens)/np.sqrt(decoder_output.shape[-1])#
		
		# Ensure that scores for padding are automatically zeroed out when all input_lengths are not same as appropriate 
		for i in range(input_length.shape[0]):
			if (torch.sum(src_ptr_scores[i,:,input_length[i]-2:]) != 0.0):
				print("max_inp_len: ", torch.max(input_length).cpu().item())
				print("src_ptr_scores.shape: ", src_ptr_scores.shape)
				print("input_length[i]: ", input_length[i])
				print("--", src_ptr_scores[i, 0, input_length[i]-2:])
			assert torch.sum(src_ptr_scores[i,:,input_length[i]-2:]) == 0.0
			
		# Make src_ptr_scores of proper shape by appending 0's for non-existent pointers
		a, b, c = src_ptr_scores.shape
		src_ptr_scores_net = torch.zeros(a, b, self.num_ptrs).cuda()
		src_ptr_scores_net[:,:,:c] = src_ptr_scores
		
		assert torch.sum(src_ptr_scores_net[:,:,c:]) == 0.0
		
		all_scores = torch.cat((tgt_vocab_scores, src_ptr_scores_net), axis=2)
		all_scores = F.log_softmax(all_scores, dim=-1) #---? dummy x, weights may not get updated effectively
		
		loss = self.loss_compute(all_scores, output_ids_y, ntokens)

		#########
		
		if not decode:
			return loss
		else:
			ys = beam_decode(input_ids, self.decoder_embed, self.decoder, encoder_output, encoder_output_excluding_sptokens,\
							 source_mask, self.generator)            
			return (loss, ys)

@dataclass(frozen=True)
class InputExample:
	"""
	A single training/test example for simple sequence classification.
	Args:
		guid: Unique id for the example.
		text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
		text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
		label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
	"""

	guid: str
	text_inp: str
	text_out: str

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
	
def sort_inp_len(inpex):
	return len(inpex.text_inp.split(" "))

class InputFeatures(object):
	"""
	A single set of features of data.
	Args:
		input_ids: Indices of input sequence tokens in the vocabulary.
		attention_mask: Mask to avoid performing attention on padding token indices.
			Mask values selected in ``[0, 1]``:
			Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
		token_type_ids: Segment token indices to indicate first and second portions of the inputs.
		label: Label corresponding to the input
	"""

	def __init__(self, input_ids, attention_mask=None, source_mask=None, \
						  token_type_ids=None, output_ids=None, \
						  output_ids_y=None, output_mask=None, ntokens=None, \
						  input_length=None, output_length=None):
		self.input_ids = input_ids
		self.attention_mask = attention_mask
		self.source_mask = source_mask
		self.token_type_ids = token_type_ids
		self.output_ids = output_ids
		self.output_ids_y = output_ids_y
		self.output_mask = output_mask
		self.ntokens = ntokens
		self.input_length = input_length
		self.output_length = output_length

	def __repr__(self):
		return str(self.to_json_string())

	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_example_from_tensor_dict(self, tensor_dict):
		"""Gets an example from a dict with tensorflow tensors
		Args:
			tensor_dict: Keys and values should match the corresponding Glue
				tensorflow_dataset examples.
		"""
		raise NotImplementedError()

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	def tfds_map(self, example):
		"""Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
		This method converts examples to the correct format."""
		if len(self.get_labels()) > 1:
			example.label = self.get_labels()[int(example.label)]
		return example


class TOPProcessor(DataProcessor):
	"""Processor for the ATIS data set (standard version)."""

	def get_example_from_tensor_dict(self, tensor_dict):
		"""See base class."""
		return InputExample(
			tensor_dict["idx"].numpy(),
			tensor_dict["input"].numpy().decode("utf-8"),
			str(tensor_dict["output"].numpy().decode("utf-8")),
		)

	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(data_dir, "train")

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(data_dir, "dev")
	
	def get_test_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(data_dir, "test")

	def get_labels(self):
		"""See base class."""
		global output_vocab
		return output_vocab
	
	def _create_examples(self, data_dir, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		input_file_data = os.path.join(data_dir, "data.tsv")
		with open(input_file_data, "r", encoding="utf-8-sig") as f:
			for i, inp in enumerate(f):
				inps = inp.split('\t')  
				guid = "%s-%s" % (set_type, i)
				text_inp = inps[1].strip()
				text_out = inps[2].strip()
				examples.append(InputExample(guid=guid, text_inp=text_inp, text_out=text_out))
				
			# Sort these out before returning
			examples = sorted(examples, key=sort_inp_len)
			return examples


def semParse_convert_examples_to_features(
	examples,
	tokenizer,
	label_list,
	max_seq_length=512,
	cls_token_at_end=False,
	cls_token="[CLS]",
	cls_token_segment_id=1,
	sep_token="[SEP]",
	sep_token_extra=False,
	pad_on_left=False,
	pad_token=0,
	pad_token_segment_id=0,
	sequence_a_segment_id=0,
	mask_padding_with_zero=True,
):
	
#     print("%%%%%%%%%%%%")
#     print(examples[:5])
	
	features = []
	
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d", ex_index, len(examples))

		tokens = []
		inp = example.text_inp.split(" ")
		for word in inp:
			tokens.append(word)

		output = example.text_out.split(" ")
		assert len(output)<=max_seq_length, "Length of output is larger than max_seq_length :: " + example.text_out
	
		# Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
		special_tokens_count = tokenizer.num_added_tokens()
		if len(tokens) > max_seq_length - special_tokens_count:
			tokens = tokens[: (max_seq_length - special_tokens_count)]
		
		tokens += [sep_token]
		output += [sep_token]
		
		if sep_token_extra:
			# roberta uses an extra separator b/w pairs of sentences
			tokens += [sep_token]
		segment_ids = [sequence_a_segment_id] * len(tokens)
		
		if cls_token_at_end:
			tokens += [cls_token]
			output += [cls_token]
			segment_ids += [cls_token_segment_id]
		else:
			tokens = [cls_token] + tokens
			output = [cls_token] + output
			segment_ids = [cls_token_segment_id] + segment_ids
			
		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		output_ids = convert_output_to_ids(output)
		
		input_length = len(input_ids)
		output_length = len(output_ids)
		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
		output_mask = [1 if mask_padding_with_zero else 0] * len(output_ids)
		

		# Zero-pad up to the sequence length.
		padding_length = max_seq_length - len(input_ids)
		padding_length_output = max_seq_length - len(output_ids)
		if pad_on_left:
			input_ids = ([pad_token] * padding_length) + input_ids
			output_ids = ([pad_token] * padding_length_output) + output_ids
			input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
			output_mask = ([0 if mask_padding_with_zero else 1] * padding_length_output) + output_mask
			segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
		else:
			input_ids += [pad_token] * padding_length
			output_ids += [pad_token] * padding_length_output
			input_mask += [0 if mask_padding_with_zero else 1] * padding_length
			output_mask += [0 if mask_padding_with_zero else 1] * padding_length_output
			segment_ids += [pad_token_segment_id] * padding_length
		
		#################
		inp_ids = torch.tensor(input_ids)
		out_ids = torch.tensor(output_ids)
		source_mask = (inp_ids != 0).unsqueeze(-2)
		trg = out_ids[:-1]
		trg_y = out_ids[1:]
		target_mask = (trg != 0).unsqueeze(-2)
		target_mask = target_mask & Variable(subsequent_mask(trg.size(-1)).type_as(target_mask.data))
		target_mask = target_mask.squeeze(0)
		
		source_mask = source_mask.data.numpy()
		output_ids_ = trg.data.numpy()
		output_ids_y = trg_y.data.numpy()
		output_mask = target_mask.data.numpy()
		ntokens = (trg_y!=0).data.sum().item()
		#################
			
		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		assert len(output_ids) == max_seq_length
		
		features.append(
			InputFeatures(input_ids=input_ids, attention_mask=input_mask, source_mask=source_mask, \
						  token_type_ids=segment_ids, output_ids=output_ids_, \
						  output_ids_y=output_ids_y, output_mask=output_mask, ntokens=ntokens, \
						  input_length=input_length, output_length=output_length)
		)

	return features


class BucketSampler(Sampler):
	def __init__(self, data_source, batch_size):
		self.data_source = data_source
		self.batch_size = batch_size
		ids = list(range(0, len(data_source)))
		self.bins = [ids[i:min(i + self.batch_size, len(ids))] for i in range(0, len(ids), batch_size)]
		
	def shuffle(self):
		np.random.shuffle(self.bins)
		
	def num_samples(self):
		return len(self.data_source)
	
	def __len__(self):
		return len(self.bins)
	
	def __iter__(self):
		return iter(self.bins)

def get_slots_info(output_ids, pred_ids):
	output_ids = output_ids.detach().cpu().numpy()
	y_slot_p = []
	y_slot_t = []
	for i in range(output_ids.shape[0]):
		slot_p = []
		slot_t = []
		for j in range(output_ids.shape[1]):
			if (output_ids[i][j]==2): break
			elif (output_ids[i][j]<num_slots+3 and output_ids[i][j]>2):
				slot_t.append(output_ids[i][j])
				if (j>=len(pred_ids[i][0])):
					print("Terminated early. Appending -1")
					print(output_ids[i][:j+5])
					print(pred_ids[i][0])
					slot_p.append(-1)
				else:
					slot_p.append(pred_ids[i][0][j])
		if (len(slot_p) > 0):
			y_slot_p += slot_p
			y_slot_t += slot_t
	return (y_slot_p, y_slot_t)

def get_intent_n_exact_match_info(output_ids, pred_ids):
	output_ids = output_ids.detach().cpu().numpy()
	y_intent_p = []
	y_intent_t = []
	exact_match_res = []
	y_intent_p_first = []
	y_intent_t_first = []
	for i in range(output_ids.shape[0]):
		intent_p = []
		intent_t = []
		y_intent_p_first.append(pred_ids[i][0][1])
		y_intent_t_first.append(output_ids[i][1])
		for j in range(output_ids.shape[1]):
			if (output_ids[i][j]==2): 
				if (np.all(pred_ids[i][0][:j+1] == output_ids[i,:j+1])): exact_match_res.append(1)
				else: exact_match_res.append(0)
				break
			elif (output_ids[i][j]<num_slots+num_intents+3 and output_ids[i][j]>=num_slots+3):
				intent_t.append(output_ids[i][j])
				if (j>=len(pred_ids[i][0])):
					print("Terminated early. Appending -1")
					print(output_ids[i][:j+5])
					print(pred_ids[i][0])
					intent_p.append(-1)
				else:
					intent_p.append(pred_ids[i][0][j])
		if (len(intent_p) > 0):
			y_intent_p += intent_p
			y_intent_t += intent_t
	return (y_intent_p, y_intent_t, exact_match_res, y_intent_p_first, y_intent_t_first)

def main():

	global outputs_map
	global output_vocab
	global all_outputs
	global num_ptrs
	global num_slots
	global num_intents

	output_vocab = ['[SL:ORGANIZER_EVENT', 'SL:METHOD_TRAVEL]', 'SL:CONTACT]', '[SL:ROAD_CONDITION_AVOID', '[SL:ATTRIBUTE_EVENT', 'SL:PATH]', 'SL:TYPE_RELATION]', '[SL:OBSTRUCTION', 'SL:DESTINATION]', 'SL:COMBINE]', '[SL:CONTACT', '[SL:NAME_EVENT', 'SL:ATTRIBUTE_EVENT]', 'SL:LOCATION_WORK]', '[SL:CATEGORY_EVENT', '[SL:WAYPOINT_ADDED', '[SL:DATE_TIME_DEPARTURE', '[SL:SEARCH_RADIUS', '[SL:PATH', 'SL:GROUP]', 'SL:LOCATION_MODIFIER]', '[SL:OBSTRUCTION_AVOID', 'SL:AMOUNT]', '[SL:ORDINAL', '[SL:CATEGORY_LOCATION', 'SL:LOCATION_USER]', 'SL:DATE_TIME_DEPARTURE]', '[SL:SOURCE', '[SL:LOCATION_CURRENT', '[SL:LOCATION_USER', '[SL:ATTENDEE_EVENT', 'SL:CONTACT_RELATED]', '[SL:LOCATION_MODIFIER', '[SL:WAYPOINT', '[SL:AMOUNT', 'SL:DATE_TIME_ARRIVAL]', 'SL:UNIT_DISTANCE]', '[SL:UNIT_DISTANCE', '[SL:WAYPOINT_AVOID', '[SL:POINT_ON_MAP', 'SL:DATE_TIME]', 'SL:LOCATION]', '[SL:DESTINATION', 'SL:SEARCH_RADIUS]', 'SL:ROAD_CONDITION_AVOID]', 'SL:WAYPOINT_AVOID]', 'SL:ORGANIZER_EVENT]', 'SL:ORDINAL]', '[SL:CONTACT_RELATED', 'SL:OBSTRUCTION_AVOID]', '[SL:LOCATION_WORK', '[SL:DATE_TIME_ARRIVAL', 'SL:ATTENDEE_EVENT]', 'SL:CATEGORY_LOCATION]', '[SL:METHOD_TRAVEL', '[SL:TYPE_RELATION', 'SL:SOURCE]', '[SL:ROAD_CONDITION', 'SL:ROAD_CONDITION]', '[SL:COMBINE', 'SL:OBSTRUCTION]', 'SL:POINT_ON_MAP]', 'SL:WAYPOINT_ADDED]', 'SL:CATEGORY_EVENT]', '[SL:DATE_TIME', '[SL:PATH_AVOID', '[SL:LOCATION', 'SL:WAYPOINT]', 'SL:LOCATION_CURRENT]', 'SL:NAME_EVENT]', '[SL:GROUP', 'SL:PATH_AVOID]']
	num_slots = len(output_vocab)
	# print("Number of slots = ", num_slots)

	output_vocab += ['IN:GET_INFO_TRAFFIC]', 'IN:GET_LOCATION_HOME]', '[IN:GET_DISTANCE', 'IN:GET_CONTACT]', '[IN:COMBINE', 'IN:COMBINE]', 'IN:GET_INFO_ROUTE]', 'IN:GET_ESTIMATED_DEPARTURE]', '[IN:UPDATE_DIRECTIONS', 'IN:GET_DIRECTIONS]', '[IN:GET_LOCATION_HOMETOWN', 'IN:GET_ESTIMATED_ARRIVAL]', '[IN:GET_CONTACT', 'IN:GET_EVENT_ATTENDEE]', '[IN:GET_EVENT_ATTENDEE_AMOUNT', '[IN:GET_ESTIMATED_DEPARTURE', 'IN:GET_LOCATION_SCHOOL]', '[IN:GET_LOCATION', 'IN:UNINTELLIGIBLE]', 'IN:GET_LOCATION_HOMETOWN]', '[IN:GET_EVENT', '[IN:GET_INFO_TRAFFIC', 'IN:GET_EVENT]', '[IN:GET_INFO_ROUTE', '[IN:GET_EVENT_ORGANIZER', 'IN:GET_LOCATION_WORK]', 'IN:GET_LOCATION]', 'IN:GET_EVENT_ATTENDEE_AMOUNT]', 'IN:NEGATION]', 'IN:UPDATE_DIRECTIONS]', 'IN:GET_DISTANCE]', '[IN:GET_ESTIMATED_DURATION', '[IN:GET_EVENT_ATTENDEE', '[IN:UNINTELLIGIBLE', '[IN:GET_LOCATION_HOME', '[IN:GET_ESTIMATED_ARRIVAL', '[IN:NEGATION', '[IN:GET_DIRECTIONS', '[IN:GET_INFO_ROAD_CONDITION', 'IN:GET_INFO_ROAD_CONDITION]', 'IN:GET_ESTIMATED_DURATION]', 'IN:GET_EVENT_ORGANIZER]', '[IN:GET_LOCATION_SCHOOL', '[IN:GET_LOCATION_WORK']
	num_intents = len(output_vocab) - num_slots
	# print("Number of labels = ", num_intents)

	ptrs = ['@ptr0', '@ptr1', '@ptr2', '@ptr3', '@ptr4', '@ptr5', '@ptr6', '@ptr7', '@ptr8', '@ptr9', '@ptr10', '@ptr11', '@ptr12', '@ptr13', '@ptr14', '@ptr15', '@ptr16', '@ptr17', '@ptr18', '@ptr19', '@ptr20', '@ptr21', '@ptr22', '@ptr23', '@ptr24', '@ptr25', '@ptr26', '@ptr27', '@ptr28', '@ptr29', '@ptr30', '@ptr31', '@ptr32', '@ptr33', '@ptr34', '@ptr35', '@ptr36', '@ptr37', '@ptr38', '@ptr39', '@ptr40', '@ptr41', '@ptr42', '@ptr43', '@ptr44', '@ptr45', '@ptr46', '@ptr47', '@ptr48', '@ptr49', '@ptr50', '@ptr51']
	num_ptrs = len(ptrs)

	output_vocab = ['PAD', '[CLS]', '[SEP]'] + output_vocab

	# print("Number of intent + slot + special tokens = ", len(output_vocab))
	# print("Number of ptrs = ", num_ptrs)

	all_outputs = output_vocab + ptrs

	# print("Number of different output tokens: ", len(all_outputs))

	outputs_map = {word: i for i, word in enumerate(all_outputs)}

	processors = {"TOP":TOPProcessor}

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	n_gpu = torch.cuda.device_count()
	# device = "cpu"
	# n_gpu = 0

	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)

	########################## Input Parameters ###########################
	task_name = "TOP"
	bert_model = "bert-base-cased"
	do_lower_case = True
	data_dir = '/mnt/nfs/scratch1/prafullpraka/696DS/TOP/TOP_BERT_tokenized/woUnsupported/train/'
	train_batch_size = 32 # 128
	per_gpu_eval_batch_size = 8 # 128
	num_train_epochs = 200
	warmup_proportion = 0.1
	learning_rate = 2e-5
	adam_epsilon = 1e-8
	weight_decay = 0.01
	local_rank = -1
	max_seq_length = 128
	max_grad_norm = 1.0
	output_dir = '/home/prafullpraka/Work/Spring2020/696DS/SubbusCode/semparse/notebooks/Exp_SemParse/output_top_bert/ptrnet_ep200padv2wU/'
	#######################################################################

	processor = processors[task_name]()
	label_list = processor.get_labels()
	num_labels = len(label_list)

	tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

	train_examples = processor.get_train_examples(data_dir)
	num_train_optimization_steps = math.ceil(
				len(train_examples) / train_batch_size) * num_train_epochs

	# ############# Changed ###########
	# ############# Hacky that passing number of slot labels separately instead of via config ############
	config = BertConfig.from_pretrained(bert_model, num_labels=num_labels, finetuning_task=task_name)
	model = BertForSequenceGenerationWithPointerNet.from_pretrained(bert_model, config = config)
	# #################################

	model.to(device)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias','LayerNorm.weight','norm.a_2', 'norm.b_2']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	warmup_steps = int(warmup_proportion * num_train_optimization_steps)
	optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

	if n_gpu > 1:
			model = torch.nn.DataParallel(model)

	if local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
														  output_device=local_rank,
														  find_unused_parameters=True)
	global_step = 0
	nb_tr_steps = 0
	tr_loss = 0
	label_map = {i : label for i, label in enumerate(label_list)}

	# ////////////////

	do_train = False
	do_eval = True
	eval_on = "dev"
	if do_eval:
		data_dir = '/mnt/nfs/scratch1/prafullpraka/696DS/TOP/TOP_BERT_tokenized/woUnsupported/test/'

	# ////////////////

	if do_train:

			## Changed ##
			train_features = semParse_convert_examples_to_features(
				examples=train_examples, tokenizer=tokenizer, label_list=label_list, \
				max_seq_length=max_seq_length, pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], 
				pad_on_left = False, pad_token_segment_id=0)

			inp_length_ = [f.input_length for f in train_features]

			#############
			logger.info("***** Running training *****")
			logger.info("  Num examples = %d", len(train_examples))
			logger.info("  Batch size = %d", train_batch_size)
			logger.info("  Num steps = %d", num_train_optimization_steps)

			########### Change ########
			all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
			all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
			all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.bool)
			all_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
			all_output_ids = torch.tensor([f.output_ids for f in train_features], dtype=torch.long)
			all_output_ids_y = torch.tensor([f.output_ids_y for f in train_features], dtype=torch.long)
			all_output_mask = torch.tensor([f.output_mask for f in train_features], dtype=torch.bool)
			all_ntokens = torch.tensor([f.ntokens for f in train_features], dtype=torch.long)
			all_input_length = torch.tensor([f.input_length for f in train_features], dtype=torch.long)
			all_output_length = torch.tensor([f.output_length for f in train_features], dtype=torch.long)
			train_data = TensorDataset(all_input_ids, all_attention_mask, all_source_mask, all_token_type_ids, \
									   all_output_ids, all_output_ids_y, all_output_mask, all_ntokens, \
									   all_input_length, all_output_length)
			###########################

			if local_rank == -1:
				train_sampler = BucketSampler(train_data, train_batch_size)

			else:
				train_sampler = DistributedSampler(train_data)

			train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

			model.train()

			for _ in trange(int(num_train_epochs), desc="Epoch"):

				logger.info("\n")
				tr_loss = 0
				nb_tr_examples, nb_tr_steps = 0, 0
				train_sampler.shuffle()

				for idx, i in enumerate(train_sampler):
					batch = train_dataloader.dataset[i]
					batch = tuple(t.to(device) for t in batch)
					input_ids, attention_mask, source_mask, token_type_ids, output_ids, \
											output_ids_y, output_mask, ntokens, input_length, output_length = batch
					outputs = model(input_ids=input_ids, attention_mask=attention_mask, source_mask=source_mask, \
									token_type_ids=token_type_ids, output_ids=output_ids, \
									output_ids_y=output_ids_y, output_mask = output_mask, ntokens=ntokens, \
									input_length=input_length, output_length=output_length)

					loss = outputs

					if (idx%100==0):
						logger.info("Loss at iteration %d = %.4f", idx, loss.item())


					############################
					if n_gpu > 1:
						loss = loss.mean() # mean() to average on multi-gpu.


					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

					tr_loss += loss.item()
					nb_tr_examples += input_ids.size(0)
					nb_tr_steps += 1

					optimizer.step()
					scheduler.step()  # Update learning rate schedule
					model.zero_grad()
					global_step += 1

				logger.info("Average loss at the end of Epoch = %.4f", tr_loss / nb_tr_steps)
				logger.info("\n")

			# Save a trained model and the associated configuration
			model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
			model_to_save.save_pretrained(output_dir)
			tokenizer.save_pretrained(output_dir)
			model_config = {"bert_model":bert_model, "do_lower":do_lower_case, "max_seq_length":max_seq_length, \
							"num_labels":len(label_list), "label_map":label_map}
			json.dump(model_config,open(os.path.join(output_dir,"model_config.json"),"w"))
			# Load a trained model and config that you have fine-tuned
	else:
		# Load a trained model and vocabulary that you have fine-tuned
		model = BertForSequenceGenerationWithPointerNet.from_pretrained(output_dir)
		tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=do_lower_case)

	model.to(device)

	if do_eval and (local_rank == -1 or torch.distributed.get_rank() == 0):
		if eval_on == "dev":
			eval_examples = processor.get_dev_examples(data_dir)
		elif eval_on == "test":
			eval_examples = processor.get_test_examples(data_dir)
		else:
			raise ValueError("eval on dev or test set only")

		eval_features = semParse_convert_examples_to_features(
			examples=eval_examples, tokenizer=tokenizer, label_list=label_list, \
			max_seq_length=max_seq_length, pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], 
			pad_on_left = False, pad_token_segment_id=0)

		logger.info("***** Running evaluation *****")
		logger.info("  Num examples = %d", len(eval_examples))
		logger.info("  Batch size = %d", per_gpu_eval_batch_size)

		all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
		all_attention_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
		all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.bool)
		all_token_type_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
		all_output_ids = torch.tensor([f.output_ids for f in eval_features], dtype=torch.long)
		all_output_ids_y = torch.tensor([f.output_ids_y for f in eval_features], dtype=torch.long)
		all_output_mask = torch.tensor([f.output_mask for f in eval_features], dtype=torch.bool)
		all_ntokens = torch.tensor([f.ntokens for f in eval_features], dtype=torch.long)
		all_input_length = torch.tensor([f.input_length for f in eval_features], dtype=torch.long)
		all_output_length = torch.tensor([f.output_length for f in eval_features], dtype=torch.long)

		eval_dataset = TensorDataset(all_input_ids, all_attention_mask, all_source_mask, all_token_type_ids, \
									 all_output_ids, all_output_ids_y, all_output_mask, all_ntokens, \
									 all_input_length, all_output_length)

		# Run prediction for full data

		eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
		eval_sampler =  BucketSampler(eval_dataset, train_batch_size)#SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)
		eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=train_batch_size)
		# multi-gpu evaluate
		if n_gpu > 1:
			model = torch.nn.DataParallel(model)

		model.eval()

		eval_loss, eval_slot_accuracy, eval_intent_accuracy = 0, 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0
		y_slot_true = []
		y_slot_pred = []
		y_intent_true = []
		y_intent_pred = []
		exact_matches = []
		y_intent_pred_first = []
		y_intent_true_first = []

		for idx, i in enumerate(tqdm(eval_sampler, desc="Evaluating")):
			logger.info("\n")
			batch = eval_dataloader.dataset[i]
			batch = tuple(t.to(device) for t in batch)

			with torch.no_grad():
				input_ids, attention_mask, source_mask, token_type_ids, output_ids, \
								output_ids_y, output_mask, ntokens, input_length, output_length = batch
				outputs = model(input_ids=input_ids, attention_mask=attention_mask, source_mask=source_mask, \
								token_type_ids=token_type_ids, output_ids=output_ids, \
								output_ids_y=output_ids_y, output_mask = output_mask, ntokens=ntokens, \
								input_length=input_length, output_length=output_length, decode=True)
				tmp_eval_loss, pred_ids = outputs

				if n_gpu > 1:
					tmp_eval_loss = tmp_eval_loss.mean()

				eval_loss += tmp_eval_loss.item()

			nb_eval_steps += 1


			y_slot_p, y_slot_t = get_slots_info(output_ids, pred_ids)
			y_intent_p, y_intent_t, em, y_intent_p_first, y_intent_t_first = get_intent_n_exact_match_info(output_ids, pred_ids)
			y_slot_pred += y_slot_p
			y_slot_true += y_slot_t
			y_intent_pred += y_intent_p
			y_intent_true += y_intent_t
			exact_matches += em
			y_intent_pred_first += y_intent_p_first
			y_intent_true_first += y_intent_t_first

		eval_loss = eval_loss / nb_eval_steps

		results = {
			"loss": eval_loss,
			"accuracy_slots": np.mean(np.array(y_slot_true) == np.array(y_slot_pred)),
			"accuracy_intents": np.mean(np.array(y_intent_true) == np.array(y_intent_pred)),
			"accuracy_intent_first": np.mean(np.array(y_intent_true_first) == np.array(y_intent_pred_first)),
			"exact_match": np.mean(np.array(exact_matches)),
		}

		logger.info("Results: ")
		logger.info(results)

if __name__ == '__main__':
	main()