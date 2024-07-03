# This is the template we are going to use for the implementation of our custom Spiking lstm cell.
# Per possibili implementazioni e aiuti guarda al seguente link: 
# https://github.com/pytorch/pytorch/blob/main/benchmarks/fastrnns/custom_lstms.py

# If you wanna check basic implementation in PyTorch:
# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py#L537

# imports
# ...
import torch
from torch import nn
import snntorch as snn

from torch import Tensor
from typing import Tuple



class SLSTM_cell(nn.Module):
    # TODO: Implement the init function in a proper way. Check also the parameters 
    def __init__(self, input_size: int, hidden_size: int, settings: dict):
        r"""__init__(input_size, hidden_size, settings)


        Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`. 
            It's the number of output values we want.
        settings: The dictionary containing all the essential parameters
            to encode the cell.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        # Setting the size of each gate population
        self.enc_pop_input = int(settings["enc_pop_input"])
        self.enc_pop_gate = int(settings["enc_pop_gate"])
        self.enc_pop_forget = int(settings["enc_pop_forget"])
        self.enc_pop_output = int(settings["enc_pop_output"])


        # We decided to implement a Linear layer so we can get rid of Weights and biases as nn.Parameters() 
        # NOTE: In some cases we have the SAME bias for the input and hidden vectors, here they are distinct
        self.Wif = nn.Linear(self.input_size, self.enc_pop_forget)
        self.Wii = nn.Linear(self.input_size, self.enc_pop_input)
        self.Wig = nn.Linear(self.input_size, self.enc_pop_gate)
        self.Wio = nn.Linear(self.input_size, self.enc_pop_output)

        self.Whf = nn.Linear(self.hidden_size, self.enc_pop_forget)
        self.Whi = nn.Linear(self.hidden_size, self.enc_pop_input)
        self.Whg = nn.Linear(self.hidden_size, self.enc_pop_gate)
        self.Who = nn.Linear(self.hidden_size, self.enc_pop_output)

        # Initialization of Leaky neurons, each population has its own beta and threshold parameter
        self.forget_gate = snn.Leaky(beta=float(settings['beta_forget']), threshold=float(settings['thr_forget']), learn_beta=True, learn_threshold=True)
        self.input_gate = snn.Leaky(beta=float(settings['beta_input']), threshold=float(settings['thr_input']), learn_beta=True, learn_threshold=True)
        self.gate_gate = snn.Leaky(beta=float(settings['beta_gate']), threshold=float(settings['thr_gate']), learn_beta=True, learn_threshold=True)
        self.output_gate = snn.Leaky(beta=float(settings['beta_output']), threshold=float(settings['thr_output']), learn_beta=True, learn_threshold=True)
        self.hidden_gate = snn.Leaky(beta=float(settings['beta_hidden']), threshold=float(settings['thr_hidden']), learn_beta=True, learn_threshold=True)
    
    
    # Function that wraps the init_leaky of every gate inside the cell
    def init_leaky(self):
        self.mem_forget = self.forget_gate.init_leaky()
        self.mem_input = self.input_gate.init_leaky()
        self.mem_gate = self.gate_gate.init_leaky()
        self.mem_output = self.output_gate.init_leaky()
        self.mem_hidden = self.hidden_gate.init_leaky()


    # TODO: Implement the forward pass
    # The block receives 4 different spiking inputs, each for the relative gate
    # In state receives (hidden_state, cell_state)
    # returns hidden and (hidden_state, cell_state)
    def forward(self, input: Tuple[Tensor, Tensor, Tensor, Tensor], state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        r"""forward(input, state)


        Args:
        input: A tuple of 4 elements, each eleemnt is a tensor of the spikes 
            coming from a different population. In order `[forget, input, gate, output]`  
        state: A tuple of 2 elements containing `hidden` and `cell state`
            It's the number of output values we want.
        """
        spk_x_1, spk_x_2, spk_x_3, spk_x_4 = input
        hidden_prev, syn_prev = state

        # FORGET GATE
        enc_forget = self.Wif(spk_x_1) + self.Whf(hidden_prev)
        spk_forget,  self.mem_forget = self.forget_gate(enc_forget, self.mem_forget)
        f = torch.sigmoid(torch.mean(spk_forget, dim=1).unsqueeze(1)) # Non abbiamo inserito l'asse, quindi di default lo fa su axis=0
        
        # INPUT GATE
        enc_input = self.Wii(spk_x_2) + self.Whi(hidden_prev)
        spk_input,  self.mem_input = self.input_gate(enc_input, self.mem_input)
        i = torch.sigmoid(torch.mean(spk_input, dim=1).unsqueeze(1)) # Non abbiamo inserito l'asse, quindi di default lo fa su axis=0
        
        # GATE GATE
        enc_gate = self.Wig(spk_x_3) + self.Whg(hidden_prev)
        spk_gate,  self.mem_gate = self.gate_gate(enc_gate, self.mem_gate)
        g = torch.tanh(torch.mean(spk_gate, dim=1).unsqueeze(1))
        
        # OUTPUT GATE
        enc_output = self.Wio(spk_x_4) + self.Who(hidden_prev)
        spk_output,  self.mem_output = self.output_gate(enc_output, self.mem_output)
        o = torch.sigmoid(torch.mean(spk_output, dim=1).unsqueeze(1)) # Non abbiamo inserito l'asse, quindi di default lo fa su axis=0
        
        syn = f * syn_prev + i * g
        mem_h = o * torch.tanh(syn)

        hidden, self.mem_hidden = self.hidden_gate(mem_h, self.mem_hidden)
        # Dal momento che stiamo passando per una popolazione di neuroni la dimensione dell'output di hidden sarà pari a enc_pop_hidden
        return hidden, (hidden, syn)


# NOTA (unica matrice 4*hidden_size come nel secondo link a inizio pagina oppure 4 diversi):
# Non sarebbe tecnicamente sbagliato avere una coppia di peso-bias per ogni gate di dimensione hidden_size. 
# Infatti, in molti casi, potrebbe essere più intuitivo e facile da gestire. Tuttavia, concatenare tutti i pesi in un unico tensore 
# di dimensione 4 * hidden_size è una convenzione comune in molte implementazioni di LSTM, compresa quella di PyTorch.

# Questa convenzione ha alcuni vantaggi. Ad esempio, può portare a un codice più pulito e compatto, poiché permette di eseguire 
# tutte le operazioni delle porte in parallelo con un’unica moltiplicazione matrice-vettore, piuttosto che eseguire quattro 
# moltiplicazioni separate. Inoltre, può anche portare a un leggero aumento delle prestazioni a causa dell’ottimizzazione del 
# calcolo matriciale.