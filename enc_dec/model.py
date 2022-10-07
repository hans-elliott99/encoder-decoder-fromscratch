import torch
import random
import string


class TranslGRU:
    def __init__(self, english_vocab_size:int, french_vocab_size:int,
                       enc_embed_dim:int, dec_embed_dim:int, 
                       hidden_size:int, max_length:int,
                       dec_type:str, 
                       stoi_mapping:dict, start_chr:str, stop_chr:str, data_level='word'
                ) -> None:
        """
        Define an Encoder-Decoder network for sequence to sequence translation.  

        Keyword arguments:\n
        vocab_size -- number of unique characters or words that appear in that data.\n
        embed_dim -- dimension of the embedded representation of each character/word.\n
        dec_type -- "simple" or "attention".\n
        hidden_size -- desired number of elements in the hidden state.\n
        max_length -- maximum length of the input/output sequence (in characters or words).\n
        stoi_mapping -- dictionary mapping from string to integer encoding.\n
        start_chr -- character or word used to start decoder.\n
        stop_chr -- character or word used to denote end-of-phrase.\n
        data_level -- is the data at the 'character' or 'word' level\n 
        """
        # Model architecture
        self.input_size = english_vocab_size
        
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.dec_type = dec_type
        self.max_length = max_length

        self.enc_hidden_size = hidden_size ## technically the dec hidden size has to be the same as the encoder rn
        self.dec_hidden_size = hidden_size ##

        self.output_size = french_vocab_size
        
        # Data processing
        self.stoi = stoi_mapping
        if data_level=='word':
            assert len(stoi_mapping.keys()) == 2, "the stoi_mapping dict must contain 2 dictionaries, the first for english words and the second for french."
            self.en_stoi = stoi_mapping[[k for k in stoi_mapping.keys()][0]]
            self.fr_stoi = stoi_mapping[[k for k in stoi_mapping.keys()][1]] 
            assert start_chr in self.en_stoi; assert start_chr in self.en_stoi
            assert stop_chr in self.fr_stoi; assert stop_chr in self.fr_stoi
        else:
            self.en_stoi = stoi_mapping
            self.fr_stoi = stoi_mapping
            assert start_chr in stoi_mapping; assert start_chr in stoi_mapping

        self.en_itos = {i:s for s, i in self.en_stoi.items()}
        self.fr_itos = {i:s for s, i in self.fr_stoi.items()}
        self.level = data_level

        self.start_chr_ix = self.en_stoi[start_chr]
        self.stop_chr_ix = self.en_stoi[stop_chr]
        assert self.start_chr_ix == self.fr_stoi[start_chr], "ensure consistent encoding of start and stop tokens"
        assert self.stop_chr_ix == self.fr_stoi[stop_chr], "ensure consistent encoding of start and stop tokens"

    def init_hidden(self) -> torch.tensor:
        """Initialize hidden state on device. Returns: hidden"""
        hidden = torch.zeros((self.enc_hidden_size, 1), device=self.device)  #(hidden size , batch size)
        return hidden

    def init_weights(self, device:torch.device, seed=123) -> None:
        """
        Randomly initialize weight tensors for the model on device. Can also be used to reset the model's weights.
        
        device -- the torch.device being used for training (all tensors will be put to this device)\n
        seed -- seeds a torch.Generator() on 'device' for reproducible weight initialization.
        """
        g = torch.Generator(device=device).manual_seed(seed)

        self.device = device

        self._init_enc_weights(g=g, device=device)
        self._init_dec_weights(g=g, device=device)
        
        for p in self.enc_params+self.dec_params:
            p.requires_grad = True

        self.n_parameters = self._count_params(self.enc_params+self.dec_params)

    def _count_params(self, params):
        return sum(p.nelement() for p in params)
    
    def restore_weights(self, enc_weights, dec_weights, device):
        """
        Restore model weights by providing the list of trained weights saved in self.enc_params and self.dec_params.  
        self.init_weights() must be run first to initialize tensors of appropriate size and to specify the device.

        weights -- the list of weights saved in self.enc_params and self.dec_params, must be in the same order as when saved\n
        """
        # first initialize random weights to initialize tensors of appropriate size
        self.init_weights(device=device) ##creates self.params & self.device

        assert len(enc_weights)==len(self.enc_params)
        assert len(dec_weights)==len(self.dec_params)
        trained_weights = [p.to(device) for p in enc_weights+dec_weights]

        for rand_p, trained_p in zip(self.enc_params+self.dec_params, trained_weights):
            rand_p.data = trained_p.data

    def _init_enc_weights(self, g, device) -> None:
        """Initialize weight tensors for the encoder."""
        concat_size = self.enc_embed_dim + self.enc_hidden_size

        self.enc_embed = torch.randn((self.input_size, self.enc_embed_dim), generator=g, device=device) * 0.1

        self.enc_W_reset = torch.randn((self.enc_hidden_size, concat_size), generator=g, device=device) * 1/concat_size**0.5 ##sigmoid
        self.enc_br = torch.randn((self.enc_hidden_size, 1), device=device) * 1/concat_size**0.5
        self.enc_W_update = torch.randn((self.enc_hidden_size, concat_size), generator=g, device=device) * 1/concat_size**0.5 ##sigmoid
        self.enc_bu = torch.randn((self.enc_hidden_size, 1), device=device) * 1/concat_size**0.5
        self.enc_W_htilde = torch.randn((self.enc_hidden_size, concat_size), generator=g, device=device) * (5/3)/concat_size**0.5 ##tanh
        #self.bh = torch.zeros((self.hidden_size, 1), device=device)
        
        self.enc_params = [self.enc_embed, self.enc_W_reset, self.enc_br, self.enc_W_update, self.enc_bu, self.enc_W_htilde]
        self.enc_param_names = ['enc_embed', 'enc_W_reset', 'enc_bias_r', 'enc_W_update', 'enc_bias_u', 'enc_W_htilde']
        
    def _init_dec_weights(self, g, device) -> None:
        """Initialize weight tensors for the decoder."""
        concat_size = self.dec_embed_dim + self.dec_hidden_size

        if self.dec_type == "attention":
            self.W_att = torch.randn((self.max_length, concat_size), generator=g, device=device) * 1/concat_size**0.5 ##softmax
            self.ba = torch.randn((self.max_length, 1), device=device) * 1/concat_size**0.5
            self.W_rel = torch.randn((self.dec_embed_dim, concat_size), generator=g, device=device) * (2/concat_size)**0.5 ##relu
            self.brel = torch.randn((self.dec_embed_dim, 1), device=device) * (2/concat_size)**0.5

            self.dec_params = [self.W_att, self.ba, self.W_rel, self.brel]
            self.dec_param_names = ['dec_W_att', 'dec_bias_a', 'dec_W_rel', 'dec_bias_rel']

        self.dec_embed = torch.randn((self.output_size, self.dec_embed_dim), generator=g, device=device) * 0.1
        self.dec_W_reset = torch.randn((self.dec_hidden_size, concat_size), generator=g, device=device) * 1/concat_size**0.5 ##sigmoid
        self.dec_br = torch.randn((self.dec_hidden_size, 1), device=device) * 1/concat_size**0.5
        self.dec_W_update = torch.randn((self.dec_hidden_size, concat_size), generator=g, device=device) * 1/concat_size**0.5 ##sigmoid
        self.dec_bu = torch.randn((self.dec_hidden_size, 1), device=device) * 1/concat_size**0.5
        self.dec_W_htilde = torch.randn((self.dec_hidden_size, concat_size), generator=g, device=device) * (5/3)/concat_size**0.5 ##tanh
        #self.bh = torch.zeros((self.hidden_size, 1), device=device)

        # FC HEAD
        self.W_h2o = torch.randn((self.output_size, self.dec_hidden_size), generator=g, device=device) * 1/self.dec_hidden_size**0.5 ##linear
        self.b_h20 = torch.randn((self.output_size, 1), device=device) * 1/self.dec_hidden_size**0.5

        self.dec_params += [self.dec_embed, self.dec_W_reset, self.dec_br, self.dec_W_update, self.dec_bu, self.dec_W_htilde, self.W_h2o, self.b_h20]
        self.dec_param_names += ['dec_embed', 'dec_W_reset', 'dec_bias_r', 'dec_W_update', 'dec_bias_u', 'dec_W_htilde', 'W_h2o', 'bias_h20']


    def encoder(self, hidden_prev, x_tensor_t):
        "ENCODER AT STEP T"
        input = self.enc_embed[x_tensor_t].unsqueeze(dim=1)
        hidden_new = self.gru(input, hidden_prev, self.enc_W_reset, self.enc_br, self.enc_W_update, self.enc_bu, self.enc_W_htilde)
        return hidden_new
    
    def simple_decoder(self, context_vector, input_char_t):
        "SIMPLE DECODER AT STEP T"
        input = self.dec_embed[input_char_t].unsqueeze(dim=1)
        hidden_new = self.gru(input, context_vector, self.dec_W_reset, self.dec_br, self.dec_W_update, self.dec_bu, self.dec_W_htilde)
        
        # run hidden state through linear layer to predict next char
        # output = torch.softmax(self.linear(hidden_new))
        logits = self.linear(hidden_new, W=self.W_h2o, b=self.b_h20)
        next_input_char = int(torch.argmax(logits, dim=0).item()) ##return just the index

        return hidden_new, logits, next_input_char
    
    def attention_decoder(self, enc_outputs, prev_hidden, input_char_t):
        "ATTENTION DECODER AT STEP T"
        # embed the current character and concatenate with previous hidden state
        embedded = self.dec_embed[input_char_t].unsqueeze(dim=1) ##shape embed_dim, 1
        concat1 = torch.cat((embedded, prev_hidden), dim=0)      ##shape embed_dim+hidden_size, 1
        # calculate attention weights by passing in the concatenation & softmaxing
        # apply the attention vector to the context vector
        atten = torch.nn.functional.softmax(self.linear(concat1, W=self.W_att, b=self.ba), dim=0).T ##shape 1, max_len. & enc_outputs have shape max_len, hidden_size
        atten_weighted = (atten @ enc_outputs).squeeze(dim=0)            ##shape hidden_size
        ## attention vector is broadcast over the entire enc output matrix (where each row is a hidden state), weighting

        # concat the attention-weighted context with the input embedding and pass through a ReLU linear layer
        concat2 = torch.cat((embedded, atten_weighted.unsqueeze(dim=1)), dim=0)         ##shape embed_dim+hidden_size, 1
        gru_input = torch.nn.functional.relu(self.linear(concat2, W=self.W_rel, b=self.brel))                 ##shape embed_dim, 1

        # pass the attention-processed inputs into the decoder gru along with previous hidden state to generate new hidden
        hidden_new = self.gru(gru_input, prev_hidden, self.dec_W_reset, self.dec_br, self.dec_W_update, self.dec_bu, self.dec_W_htilde)

        # run hidden state through linear layer to predict next char
        logits = self.linear(hidden_new, W=self.W_h2o, b=self.b_h20)
        next_input_char = int(torch.argmax(logits, dim=0).item())
        
        return hidden_new, logits, next_input_char
    
    def linear(self, input, W, b):
        """Wx+b"""
        return W @ input + b
    
    def gru(self, input, hidden,
                  W_reset, br,
                  W_update, bu,
                  W_htilde
                  ) -> torch.tensor:
        """One forward step in a GRU cell to update hidden state. Returns: hidden_new"""
        # Concatenate inputs with incoming hidden state
        concat_raw = torch.cat((input, hidden), dim=0)

        # Calc reset gate and apply to hidden state to produce gated/reset hidden state
        r_gate = torch.sigmoid(W_reset @ concat_raw + br)
        hidden_reset = hidden * r_gate

        # Concatenate inputs with gated hidden state
        concat_gated = torch.cat((input, hidden_reset), dim=0)
        # Calculate h tilde, the proposed new hidden state, using the gated concatenation
        h_tilde = torch.tanh(W_htilde @ concat_gated)

        # Calc update gate using the raw/ungated concatenation
        u_gate = torch.sigmoid(W_update @ concat_raw + bu)

        # Update hidden state with (1 - update gate) * hidden_t-1 + (update gate * h tilde):
        hidden_new = (1 - u_gate) * hidden + u_gate * h_tilde
        
        return hidden_new

    def backprop_update(self, enc_optimizer, dec_optimizer) -> None:
        """Zero gradients, backpropogate via loss, and update params with optimizer"""
        # zero grads
        enc_optimizer.zero_grad(set_to_none=True)
        dec_optimizer.zero_grad(set_to_none=True)
        # backprop
        self.loss.backward()
        # update
        enc_optimizer.step()
        dec_optimizer.step()

    def forward(self, x:torch.tensor, y:torch.tensor=None, teacher_forcing_ratio=0.0, max_length=100):
        """
        Perform an entire forward pass of one sample to calculate outputs and calculate loss if y is provided.
        Returns output_chars, loss.
        """

        # Determine maximum length to predict for
        if y is None:
            target_length = max_length
        else:
            target_length = y.shape[0]
            # y_ohe = torch.nn.functional.one_hot(y, num_classes=self.output_size)


        # ENCODER
        # Tensor to store the encoder gru's outputs at each timestep (as columns)
        enc_outputs = torch.zeros((self.max_length, self.enc_hidden_size), device=self.device)
        # Initialize hidden state with zeros
        enc_hidden = self.init_hidden()

        # Pass through X sequentially into the encoder and update hidden state
        loss = 0
        for t in range(x.shape[0]):
            enc_hidden = self.encoder(hidden_prev=enc_hidden, x_tensor_t=x[t])
            enc_outputs[t] = enc_hidden.squeeze(1)
        
        # encoder passes the finall hidden state (context vector) to simple decoder or attention decoder
        # the attention decoder also uses the enc_outputs matrix (ie, it uses all of the hidden states produced in the loop above)

        # DECODER
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False ##returns True if the float between [0, 1] is less than ratio
        output_chrs = []
        dec_hidden = enc_hidden ##last enc hidden state is the first context vector for simple decoder/first hidden state for attention decoder

        # Use the final hidden state (context vector) and the start token to start the decoding and predict the next character in the sequence
        # Repeat until the stop token is predicted or the target length is reached
        input_chr = self.start_chr_ix 
        for t in range(target_length):
            # SIMPLE
            if self.dec_type=="simple":
                # The last encoder hidden vector is 
                dec_hidden, logits, input_chr = self.simple_decoder(context_vector=dec_hidden, input_char_t=input_chr)
                output_chrs.append(input_chr)
            
            # ATTENTION
            if self.dec_type=="attention":
                dec_hidden, logits, input_chr = self.attention_decoder(enc_outputs=enc_outputs, 
                                                                       prev_hidden=dec_hidden,
                                                                       input_char_t=input_chr)  ##returns predicted character
                output_chrs.append(input_chr)

            if y is not None:
                loss += torch.nn.functional.cross_entropy(logits.squeeze(), y[t])

            if input_chr == self.stop_chr_ix:
                break

            if use_teacher_forcing:
                # overwrite the predicted input_chr since we will give the model the true input chr
                input_chr = y[t].item()
        
        # Save final loss for backpropagation through time
        self.loss = loss

        return output_chrs, loss
    
    @torch.no_grad()
    def translate(self, english_text:str="hello"):
        """Forward passes english_txt + 'stop_chr' through the model. level is either 'word' or 'character'."""
        remove_punct = ''.join([p for p in string.punctuation if p not in "'-<"])

        if self.level=="word":
            text = english_text+" <"
            text = ''.join([c for c in text.lower() if c not in remove_punct])
            for word in text.split():
                assert word in self.en_stoi, f"ensure all words are in the stoi_mapping ({word} is not)"
            SPACES = True
            input = torch.tensor([self.en_stoi[w] for w in text.split()], device=self.device)

        elif self.level=="character":
            text = english_text+"<"
            for c in text:
                assert c in self.en_stoi, f"ensure all characters are in the stoi_mapping. ({c} is not)"
            SPACES=False
            input = torch.tensor([self.en_stoi[c] for c in text], device=self.device)

        for _ in range(input.shape[0]):
            output, l = self.forward(input, y=None)

        pred_transl, _ = self.decode_output(output,spaces=SPACES)
         
        # if check_google:
        #     try:
        #         translator = google_translator()
        #         google_transl = translator.translate(english_text, lang_src='en', lang_tgt='fr')
        #     except:
        #         google_transl = "Error accessing Google translate."

        return ''.join([c for c in pred_transl if c not in self.fr_itos[self.stop_chr_ix]])

    def decode_output(self, output, label=None, spaces=False):
        """Returns predicted translation, true translation ( = None if label is None)"""
        pred = []
        truth = [] if label is not None else None
        
        for i in range(len(output)):
            pred.append(self.fr_itos[output[i]])
        if label is not None:
            for i in range(len(label)):
                truth.append(self.fr_itos[label[i].item()])

        if spaces:
            pred_transl = " ".join(pred)
            true_transl = " ".join(truth) if label is not None else None
        else:
            pred_transl = "".join(pred)
            true_transl = "".join(truth) if label is not None else None
        return pred_transl, true_transl