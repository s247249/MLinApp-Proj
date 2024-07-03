import torch.nn as nn
import torch
import snntorch as snn
from SLSTM import SLSTM_cell 

class SLSTM(nn.Module) :
    #TODO: implement the wrapper with the needed parameters
    def __init__(self, input_size: int, hidden_size: int, num_steps: int, num_class: int, settings: dict):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_steps = num_steps

        ##### Initialize layers #####
        ### Encoding layer
        self.fc_input = nn.Linear(input_size, int(settings["enc_pop"]))

        # One encoding popuation for each gate (4 in total)
        # Note: might consider to use different parameters for the different populations
        self.s_enc_f = snn.Leaky(beta=float(settings['beta_enc']), threshold=float(settings['thr_enc']), learn_beta=True, learn_threshold=True)  
        self.s_enc_i = snn.Leaky(beta=float(settings['beta_enc']), threshold=float(settings['thr_enc']), learn_beta=True, learn_threshold=True)  
        self.s_enc_g = snn.Leaky(beta=float(settings['beta_enc']), threshold=float(settings['thr_enc']), learn_beta=True, learn_threshold=True)
        self.s_enc_o = snn.Leaky(beta=float(settings['beta_enc']), threshold=float(settings['thr_enc']), learn_beta=True, learn_threshold=True)

        ### SLSTM cell
        self.slstm = SLSTM_cell(int(settings["enc_pop"]), hidden_size, settings) # La nuova input_size sarà pari a enc_pop
        
        ### Output layer per la classificazione
        self.fc_output = nn.Linear(hidden_size, num_class)
        self.s_output = snn.Leaky(beta=float(settings['beta_out']), threshold=float(settings['thr_out']), learn_beta=True, learn_threshold=True)


    # TODO: implement the forward function with the last classification layer
    #Input args:
    # x : input tensor of shape (B,T,C): B = batch, T = timesteps(40), C = channels(6) -> input_size
    #Output:
    #...
    def forward(self, x):

        # Inizializzazione delle membrane
        mem_f = self.s_enc_f.init_leaky()
        mem_i = self.s_enc_i.init_leaky()
        mem_g = self.s_enc_g.init_leaky()
        mem_o = self.s_enc_o.init_leaky()
        mem_h = self.s_output.init_leaky()
        self.slstm.init_leaky() # Non ci prendiamo le mem perché vengono salvate all'interno della cella (guarda la init_leaky() di SLSTM_cell)

        # Inizializzazione di hidden e cell state
        hidden_state = torch.zeros((x.shape[0], self.hidden_size))
        cell_state = torch.zeros((x.shape[0], self.hidden_size))
        state = (hidden_state, cell_state) # Ordine (hidden, syn) in slstm cell

        spk_rec = []
        mem_rec = []
        
        # Iterate over the time steps
        for step in range(self.num_steps):
            cur_enc = self.fc_input(x[:, step]) # x[:, step] -> (batch_size, 1, n_channels o input_size)

            spk_f, mem_f = self.s_enc_f(cur_enc, mem_f)
            spk_i, mem_i = self.s_enc_i(cur_enc, mem_i)
            spk_g, mem_g = self.s_enc_g(cur_enc, mem_g)
            spk_o, mem_o = self.s_enc_o(cur_enc, mem_o)

            # input: Tuple[Tensor, Tensor, Tensor, Tensor], state: Tuple[Tensor, Tensor]
            hidden, state = self.slstm(input=(spk_f, spk_i, spk_g, spk_o), state=state)
            
            cur_output = self.fc_output(hidden)
            output, mem_h = self.s_output(cur_output, mem_h)

            spk_rec.append(output)
            mem_rec.append(mem_h)
        
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
      



# num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two SLSTMs together to form a `stacked SLASTM`,
#             with the second SLSTM taking in outputs of the first SLSTM and
#             computing the final results. Default: 1