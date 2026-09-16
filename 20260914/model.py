import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
	"""将中文词语序列编码为隐藏状态和上下文表示。

	输入 source 的形状为 (batch_size, source_length)
	Encoder 输出：
	    outputs: 每个中文词对应的隐藏表示，形状为
	        (batch_size, source_length, hidden_size)
	    hidden: 最后一个时间步的隐藏状态，形状为
	        (1, batch_size, hidden_size)
	"""
	def __init__(self, input_size, embedding_size=256, hidden_size=256, dropout=0.3):
        
		super().__init__()
		# 将中文词编号转换为稠密向量。
		self.embedding = nn.Embedding(input_size, embedding_size)
		self.dropout = nn.Dropout(dropout)
		# batch_first=True 表示输入形状使用 (batch, sequence, feature)。
		self.rnn = nn.GRU(
			embedding_size,
			hidden_size,
			batch_first=True
		)

	def forward(self, source, source_lengths=None):
		# source: (batch_size, source_length)
		embedded = self.dropout(self.embedding(source))
		# embedded: (batch_size, source_length, embedding_size)
		if source_lengths is None:
			outputs, hidden = self.rnn(embedded)
		else:
			packed = pack_padded_sequence(
				embedded,
				source_lengths.cpu(),
				batch_first=True,
				enforce_sorted=False
			)
			packed_outputs, hidden = self.rnn(packed)
			outputs, _ = pad_packed_sequence(
				packed_outputs,
				batch_first=True,
				total_length=source.size(1)
			)
		return outputs, hidden


class Decoder(nn.Module):
	"""根据 Encoder 的结果逐步生成英文词语。

	输入 target 的形状为 (batch_size, target_length)，通常是英文
	序列右移后的输入，例如 <SOS> I am。每次 forward 返回每个
	位置上预测下一个英文词的 logits。
	"""
	def __init__(self, output_size, embedding_size=256, hidden_size=256, dropout=0.3):
		super().__init__()
		# output_size 是英文词表大小。
		self.embedding = nn.Embedding(output_size, embedding_size)
		self.dropout = nn.Dropout(dropout)
		self.rnn = nn.GRU(
			embedding_size + hidden_size,   # 输入词向量与上下文向量拼接后的维度。
			hidden_size,
			batch_first=True
		)
		# 将 GRU 的隐藏表示映射为英文词表中每个词的分数。
		self.output = nn.Linear(hidden_size, output_size)

	def forward(self, target, hidden, encoder_outputs, source_lengths=None):
		# target: (batch_size, target_length)
		embedded = self.dropout(self.embedding(target))
		# embedded: (batch_size, target_length, embedding_size)

		if source_lengths is None:
			context = encoder_outputs[:, -1:, :]
		else:
			last_indices = (source_lengths - 1).to(encoder_outputs.device)
			context = encoder_outputs[
				torch.arange(encoder_outputs.size(0), device=encoder_outputs.device),
				last_indices
			].unsqueeze(1)
		# 将上下文复制到目标序列的每个时间步。
		context = context.expand(-1, embedded.size(1), -1)
		rnn_input = torch.cat((embedded, context), dim=-1)

		# hidden 初始值来自 Encoder，之后使用 Decoder 返回的新 hidden。
		outputs, hidden = self.rnn(rnn_input, hidden)
		outputs = self.dropout(outputs)
		# logits: (batch_size, target_length, output_size)
		logits = self.output(outputs)
		return logits, hidden
