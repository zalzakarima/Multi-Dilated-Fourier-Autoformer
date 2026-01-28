import  torch
import  torch.nn as nn
from    model.layers.Embed import DataEmbedding_wo_pos, TokenEmbedding
from    model.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from    model.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from    model.layers.Autoformer_EncDec_multi import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi, DFT_series_decomp

class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        
        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)
    

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = TokenEmbedding(configs.enc_in, configs.d_model)
        self.dec_embedding = TokenEmbedding(configs.dec_in, configs.d_model)
        # self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                           configs.dropout)
        # self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                           configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        FourierBlock(in_channels=configs.d_model,
                                    out_channels=configs.d_model,
                                    seq_len=self.seq_len,
                                    modes=self.modes,
                                    mode_select_method=self.mode_select),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        FourierBlock(in_channels=configs.d_model,
                                    out_channels=configs.d_model,
                                    seq_len=self.seq_len // 2 + self.pred_len,
                                    modes=self.modes,
                                    mode_select_method=self.mode_select),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        FourierCrossAttention(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len_q=self.seq_len // 2 + self.pred_len,
                                            seq_len_kv=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select,
                                            num_heads=configs.n_heads),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        # print('before-enc:', x_enc.shape)
        # print('-------------------------')
        enc_out = self.enc_embedding(x_enc)
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print('input-enc:', enc_out.shape)
        # print('-------------------------')
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print(enc_out.shape)
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        # dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        
        # print(dec_out[:, -self.pred_len:, -1].shape)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, -1], attns
        else:
            return dec_out[:, -self.pred_len:, -1]  # [B, L, D]