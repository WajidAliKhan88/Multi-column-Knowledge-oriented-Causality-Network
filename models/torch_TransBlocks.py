import torch
from torch import nn
from modules.torch_transformer import Encoder, PositionalEncoding
from time import time

class TB(nn.Module):
    def __init__(self, token_embeddings, args, logger):
        super(TB, self).__init__()
        start_t = time()
        n_dict, n_emb = token_embeddings.shape
        self.att_hidden = args.n_emb
        self.gru_hidden = args.n_hidden
        self.crn_hidden = 4 * args.n_hidden
        self.max_len = args.max_len['full']
        self.n_block = args.n_block           # n_block mean layers...
        self.n_head = args.n_head
        self.is_sinusoid = args.is_sinusoid
        self.word_embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        self.pos_embedding = PositionalEncoding(n_emb, max_len=self.max_len)
        self.emb_dropout = nn.Dropout(args.dropout['emb'])
    # Transformer network
        self.transformer = Encoder(self.n_head, self.n_block, n_emb, args.dropout['layer'])

# The whole neural network in one line...
        self.out_fc = nn.Sequential(nn.Linear(self.max_len * self.att_hidden, self.att_hidden),
                                    nn.ReLU(),
                                    nn.Dropout(args.dropout['layer']),
                                    nn.Linear(self.att_hidden, args.n_class))
# initiate the weights...
        self._init_weights(token_embeddings)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def _init_weights(self, embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

    def forward(self, x, x_pre, x_alt, x_cur, seq_lens):
        x_mask = (x != 0).unsqueeze(-2)

    # Word embedding...
        x_word_emb = self.word_embedding(x)

    # Position embedding...
        x_word_emb = self.pos_embedding(x_word_emb)

    # Apply dropout...
        x_embeded = self.emb_dropout(x_word_emb)

    # produce y_transformed by passing x_embeded and x_mask into transformer...
        y_transformed = self.transformer(x_embeded, x_mask)

    # produce y_word by reshaping y_transformed ...
        y_word = torch.reshape(y_transformed, [-1, self.max_len * self.att_hidden])

    # return output by passing y_word into out_fc layer...
        return self.out_fc(y_word)
