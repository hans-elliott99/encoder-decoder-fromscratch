import torch
import numpy as np
import random
import string



class TranslGRU:
    def __init__(self, vocab_size, 
                       enc_embed_dim, dec_embed_dim,
                       dec_type, hidden_size, max_length,
                       stoi_mapping, start_chr, stop_chr
                ) -> None:
        """
        Define an Encoder-Decoder network each consisting of a GRU for sequence to sequence translation.
        n_letters = number of unique letters/characters that appear in the data.
        embed_dim = dimension of the embedding layer
        dec_type = "simple" or "attention"  
        hidden_size = desired number of elements in the hidden and cell states
        output_size = Dense layer ontop of the decoder converts last hidden state to logits of shape (output_size, 1)
        """
        # Model architecture
        self.input_size = vocab_size
        
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.dec_type = dec_type
        self.max_length = max_length

        self.enc_hidden_size = hidden_size ## technically the dec hidden size has to be the same as the encoder rn
        self.dec_hidden_size = hidden_size ##

        self.output_size = vocab_size
        
        # Data processing
        self.stoi = stoi_mapping
        self.itos = {i:s for s, i in stoi_mapping.items()}

        assert start_chr in stoi_mapping
        assert stop_chr in stoi_mapping
        self.start_chr_ix = stoi_mapping[start_chr]
        self.stop_chr_ix = stoi_mapping[stop_chr]

    def init_hidden(self) -> torch.tensor:
        """Initialize hidden state. Returns: hidden"""
        hidden = torch.zeros((self.enc_hidden_size, 1), device=self.device)  #(hidden size , batch size)
        return hidden

    def init_weights(self, device, seed=123) -> None:
        """Initialize weight tensors for the model."""
        g = torch.Generator(device=device).manual_seed(seed)

        self.device = device
        self.params = []
        self.param_names = []

        self._init_enc_weights(g=g, device=device)
        self._init_dec_weights(g=g, device=device)
        
        for p in self.params:
            p.requires_grad = True

        self.n_parameters = self._count_params()

    def _init_enc_weights(self, g, device) -> None:
        """Initialize weight tensors for the decoder model."""
        std = 1.0 / np.sqrt(self.enc_hidden_size)
        concat_size = self.enc_embed_dim + self.enc_hidden_size

        self.enc_embed = (-std - std) * torch.rand((self.input_size, self.enc_embed_dim), generator=g, device=device) + std

        self.enc_W_reset = (-std - std) * torch.rand((self.enc_hidden_size, concat_size), generator=g, device=device) + std
        self.enc_br = torch.zeros((self.enc_hidden_size, 1), device=device)
        self.enc_W_update = (-std - std) * torch.rand((self.enc_hidden_size, concat_size), generator=g, device=device) + std
        self.enc_bu = torch.zeros((self.enc_hidden_size, 1), device=device)
        self.enc_W_htilde = (-std - std) * torch.rand((self.enc_hidden_size, concat_size), generator=g, device=device) + std
        #self.bh = torch.zeros((self.hidden_size, 1), device=device)
        
        self.params += [self.enc_embed, self.enc_W_reset, self.enc_br, self.enc_W_update, self.enc_bu, self.enc_W_htilde]
        self.param_names += ['enc_embed', 'enc_W_reset', 'enc_br', 'enc_W_update', 'enc_bu', 'enc_W_htilde']
        
    def _init_dec_weights(self, g, device) -> None:
        """Initialize weight tensors for the decoder."""
        std = 1.0 / np.sqrt(self.dec_hidden_size)
        concat_size = self.dec_embed_dim + self.dec_hidden_size

        if self.dec_type == "attention":
            self.W_att = (-std - std) * torch.rand((self.max_length, concat_size), generator=g, device=device) + std
            self.ba = torch.zeros((self.max_length, 1), device=device)
            self.W_rel = (-std - std) * torch.rand((self.dec_embed_dim, concat_size), generator=g, device=device) + std
            self.brel = torch.zeros((self.dec_embed_dim, 1), device=device)

            self.params += [self.W_att, self.ba, self.W_rel, self.brel]
            self.param_names += ['W_att', 'ba', 'W_rel', 'brel']

        self.dec_embed = (-std - std) * torch.rand((self.input_size, self.dec_embed_dim), generator=g, device=device) + std
        self.dec_W_reset = (-std - std) * torch.rand((self.dec_hidden_size, concat_size), generator=g, device=device) + std
        self.dec_br = torch.zeros((self.dec_hidden_size, 1), device=device)
        self.dec_W_update = (-std - std) * torch.rand((self.dec_hidden_size, concat_size), generator=g, device=device) + std
        self.dec_bu = torch.zeros((self.dec_hidden_size, 1), device=device)
        self.dec_W_htilde = (-std - std) * torch.rand((self.dec_hidden_size, concat_size), generator=g, device=device) + std
        #self.bh = torch.zeros((self.hidden_size, 1), device=device)
        # FC HEAD
        self.W_h2o = (-std - std) * torch.rand((self.output_size, self.dec_hidden_size), generator=g, device=device) + std
        self.b_h20 = torch.zeros((self.output_size, 1), device=device)

        self.params += [self.dec_embed, self.dec_W_reset, self.dec_br, self.dec_W_update, self.dec_bu, self.dec_W_htilde, self.W_h2o, self.b_h20]
        self.param_names += ['dec_embed', 'dec_W_reset', 'dec_br', 'dec_W_update', 'dec_bu', 'dec_W_htilde', 'W_h2o', 'b_h20']


    def encoder(self, hidden_prev, x_tensor_t):
        "ENCODER AT STEP T"
        input = self.enc_embed[x_tensor_t].unsqueeze(dim=1)
        hidden_new = self.gru(input, hidden_prev, self.enc_W_reset, self.enc_br, self.enc_W_update, self.enc_bu, self.enc_W_htilde)
        return hidden_new
    
    def simple_decoder(self, context_vector, input_char_t):
        "DECODER AT STEP T"
        input = self.dec_embed[input_char_t].unsqueeze(dim=1)
        hidden_new = self.gru(input, context_vector, self.dec_W_reset, self.dec_br, self.dec_W_update, self.dec_bu, self.dec_W_htilde)
        
        # run hidden state through linear layer to predict next char
        # output = torch.softmax(self.linear(hidden_new))
        logits = self.linear_head(hidden_new)
        next_input_char = int(torch.argmax(logits, dim=0).item()) ##return just the index

        return hidden_new, logits, next_input_char
    
    def attention_decoder(self, enc_outputs, prev_hidden, input_char_t):
        "ATTENTION DECODER AT STEP T"
        # embed the current character and concatenate with previous hidden state
        embedded = self.dec_embed[input_char_t].unsqueeze(dim=1) ##shape embed_dim, 1
        concat1 = torch.cat((embedded, prev_hidden), dim=0).to(self.device)      ##shape embed_dim+hidden_size, 1
        # calculate attention weights by passing in the concatenation & softmaxing
        # apply the attention vector to the context vector
        atten = torch.nn.functional.softmax(self.linear_attention(concat1), dim=0).T ##shape 1, max_len. & enc_outputs have shape max_len, hidden_size
        atten_weighted = (atten @ enc_outputs).squeeze(dim=0)            ##shape hidden_size
        ## attention vector is broadcast over the entire enc output matrix (where each row is a hidden state), weighting

        # concat the attention-weighted context with the input embedding and pass through a ReLU linear layer
        concat2 = torch.cat((embedded, atten_weighted.unsqueeze(dim=1)), dim=0).to(self.device)          ##shape embed_dim+hidden_size, 1
        gru_input = torch.nn.functional.relu(self.linear_relu(concat2))                 ##shape embed_dim, 1

        # pass the attention-processed inputs into the decoder gru along with previous hidden state to generate new hidden
        hidden_new = self.gru(gru_input, prev_hidden, self.dec_W_reset, self.dec_br, self.dec_W_update, self.dec_bu, self.dec_W_htilde)

        # run hidden state through linear layer to predict next char
        logits = self.linear_head(hidden_new)  
        next_input_char = int(torch.argmax(logits, dim=0).item())
        
        return hidden_new, logits, next_input_char
    
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
    
    def linear_head(self, input):
        return self.W_h2o @ input + self.b_h20

    def linear_attention(self, input):
        return self.W_att @ input + self.ba

    def linear_relu(self, input):
        return self.W_rel @ input + self.brel

    def backprop_update(self, optimizer) -> None:
        """Zero gradients, backpropogate via loss, and update params with optimizer"""
        # ensure gradients are zerod
        # for p in self.params:
        #     p.grad = None
        optimizer.zero_grad(set_to_none=True)
        # backprop
        self.loss.backward()

        # update
        # for i, p in enumerate(self.params):
        #     p.data += -lr * p.grad
        optimizer.step()

    def _count_params(self):
        n_params = 0
        for p in self.params:
            n_params += p.nelement()
        return n_params


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
    
    def evaluate(self, english_text:str="hello", level="word"):
        """Forward passes english_txt + 'stop_chr' through the model. level is either 'word' or 'character'."""
        remove_punct = ''.join([p for p in string.punctuation if p not in "'<"])

        if level=="word":
            text = english_text+" <"
            text = ''.join([c for c in text.lower() if c not in remove_punct])
            input = torch.tensor([self.stoi[w] for w in text.split()], device=self.device)
        elif level=="character":
            text = english_text+"<"
            input = torch.tensor([self.stoi[c] for c in text], device=self.device)

        with torch.no_grad():
            for t in range(input.shape[0]):
                output, loss = self.forward(input, y=None)

        spaces=True
        if level=='character': spaces=False
        pred_transl, _ = self.decode_output(output,spaces=spaces)

        # if check_google:
        #     try:
        #         translator = google_translator()
        #         google_transl = translator.translate(english_text, lang_src='en', lang_tgt='fr')
        #     except:
        #         google_transl = "Error accessing Google translate."
        #     return pred_transl, google_transl
        # else:

        return pred_transl

    def decode_output(self, output, label=None, spaces=False):
        """Returns predicted translation, true translation ( = None if label is None)"""
        pred = []
        truth = [] if label is not None else None
        
        # input = [itos[ix.item()] for ix in x]
        for i in range(len(output)):
            pred.append(self.itos[output[i]])
        if label is not None:
            for i in range(len(label)):
                truth.append(self.itos[label[i].item()])

        if spaces:
            pred_transl = " ".join(pred)
            true_transl = " ".join(truth) if label is not None else None
        else:
            pred_transl = "".join(pred)
            true_transl = "".join(truth) if label is not None else None
        return pred_transl, true_transl