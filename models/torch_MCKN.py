import torch
from torch import nn
from torch.nn import utils as nn_utils
from modules.torch_transformer import PositionalEncoding
from modules.torch_TextCNNNet import TextCNNNet
from time import time


class MCKN(nn.Module):
    def __init__(self, token_embeddings, args, logger):
        super(MCKN, self).__init__()
        start_t = time()
        n_dict, n_emb = token_embeddings.shape   # shape mean size...
        self.max_len = args.max_len['full']
        self.att_hidden = n_emb
        #self.gru_hidden = args.n_hidden
        self.crn_hidden = 4 * args.n_hidden   # crn mean Causality Relation Network...
        self.n_layer = args.n_layer
        self.n_filter = args.n_filter
        self.n_kernels = args.n_kernels
        self.is_sinusoid = args.is_sinusoid
        self.is_ffn = args.is_ffn

        self.word_embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        if self.is_sinusoid:
            self.position_embedding = PositionalEncoding(n_emb, max_len=self.max_len)
        self.emb_dropout = nn.Dropout(args.dropout['emb'])



    # TextCNNNet (mean 3 column k-oriented net) of three level for pre,alt, and cur segments..
        self.pre_encoder = TextCNNNet(n_emb, args.max_len['pre'], self.n_filter, self.n_kernels)
        self.alt_encoder = TextCNNNet(n_emb, args.max_len['alt'], self.n_filter, self.n_kernels)
        self.cur_encoder = TextCNNNet(n_emb, args.max_len['cur'], self.n_filter, self.n_kernels)

    # The g_fc is the first function of RN(Relation Network) in one line...
        self.g_fc = nn.Sequential(nn.Linear(6 * self.n_filter + 2 * args.n_hidden, self.crn_hidden),
                                  nn.ReLU(),
                                  nn.Dropout(args.dropout['layer']),
                                  nn.Linear(self.crn_hidden, self.crn_hidden),
                                  nn.ReLU())

    # The f_fc is the second function of RN(Relation Network) in one line...
        self.f_fc = nn.Sequential(nn.Linear(self.crn_hidden, self.crn_hidden),
                                  nn.ReLU(),
                                  nn.Dropout(args.dropout['layer']))

    # The Output fully connected neural network in one line...
        self.out_fc = nn.Sequential(nn.Linear(self.crn_hidden, self.gru_hidden),
                                    nn.ReLU(),
                                    nn.Dropout(args.dropout['layer']),
                                    nn.Linear(self.gru_hidden, args.n_class))

        self._init_weights(token_embeddings)

        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def _init_weights(self, embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

    def forward(self, x, x_pre, x_alt, x_cur, seq_lens):
        batch_size = x.shape[0]
        sorted_seq_lens, indices = torch.sort(seq_lens, dim=0, descending=True)

        _, desorted_indices = torch.sort(indices, descending=False)
        x = x[indices]

    # Word and Segment embedding...
        x_word_emb = self.word_embedding(x)
        x_pre_word_emb = self.word_embedding(x_pre)
        x_alt_word_emb = self.word_embedding(x_alt)
        x_cur_word_emb = self.word_embedding(x_cur)

    # Apply dropout on each segment...
        x_pre_word_emb = self.emb_dropout(x_pre_word_emb)
        x_alt_word_emb = self.emb_dropout(x_alt_word_emb)
        x_cur_word_emb = self.emb_dropout(x_cur_word_emb)

        if self.is_sinusoid:
    #       x_word_emb = x_word_emb + self.position_embedding(x_word_emb), and then apply dropout emb_dropout(x_Word_emb)..
            x_word_emb += self.position_embedding(x_word_emb)
        x_word_emb = self.emb_dropout(x_word_emb)

    # apply rnn.pack_padded_seq on the input x_word_emb...
        x_word_emb = nn_utils.rnn.pack_padded_sequence(x_word_emb, sorted_seq_lens, batch_first=True)

    # Create the y_pre, y_alt, and y_cur by passing them to TextCNNNet(mean 3 column k-oriented net) encoder network
        y_pre = self.pre_encoder(x_pre_word_emb)
        y_alt = self.alt_encoder(x_alt_word_emb)
        y_cur = self.cur_encoder(x_cur_word_emb)

    # Construct four object pair by Concatenating Relation b/w pre-cur, cur-pre, pre-alt, and alt-cur...
        pre_cur = torch.cat((y_pre, y_cur), dim=1)
        cur_pre = torch.cat((y_cur, y_pre), dim=1)
        pre_alt = torch.cat((y_pre, y_alt), dim=1)
        alt_cur = torch.cat((y_alt, y_cur), dim=1)

    # Make stack of all relations pairs...
        y_composed = torch.stack([pre_cur, cur_pre, pre_alt, alt_cur], dim=1)
        y_state = torch.unsqueeze(y_state, 1)
        y_state = y_state.repeat(1, 4, 1)   # from 1 to 4 increase by 1.....

    # Concatenate y_composed and y_state...and reshape it by using view function...y_pair = OP in paper eq.(7),in paper.
    # Reshape...
        y_pair = torch.cat([y_composed, y_state], 2)
        y_pair = y_pair.view(batch_size * 4, 6 * self.n_filter + 2 * self.gru_hidden)

    # pass y_pair (OP) to g_fc function of Causality Relation Network(CRN)...
        y_pair = self.g_fc(y_pair)

    # Change the shape of y_pair (OP)...
        y_pair = y_pair.view(batch_size, 4, self.crn_hidden)

    # Sum y_pair in dim 1...
        y_pair = y_pair.sum(1).squeeze()

    # Produce y_segment  by passing OP (y_pair) to f_fc function of Causality Relation Network(CRN)...
        y_segment = self.f_fc(y_pair)

    # This out_fully connected layer take y_segment as an input..Out_fc = Final_rep..shown in eq. (8), in the paper.
        return self.out_fc(y_segment)
