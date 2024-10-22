import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import datetime
# import time
# import matplotlib.pyplot as plt
# from tqdm import tqdm
import math
import copy

import orbitFunctions


class FCQ(nn.Module): # taken from Grokking
    def __init__(self,input_dim=9,output_dim=1,hidden_dims=(10,10),activation_fc=F.relu):
        super(FCQ,self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim,hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for ii in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[ii],hidden_dims[ii+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1],output_dim)


    def forward(self,state):
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32)#,device=self.device)
            x = x.unsqueeze(0)
        if x.shape[-1] == 1:
            x = torch.reshape(x,[x.shape[0],x.shape[2],x.shape[1]])
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x


class CNN1D(nn.Module): # need to verify this works
    def __init__(self,input_dim,output_dim,hidden_dims=(10,10),activation_fc=F.relu):
        super(CNN1D,self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim,hidden_dims[0])
        # self.hidden_layers = nn.ModuleList()
        self.c1 = nn.Conv1d(1,5,3)
        self.c2 = nn.Conv1d(5,5,3)
        self.c3 = nn.Conv1d(5,7,3)
        # self.fc1 = nn.Linear(6,13) # need to work on this layer
        # self.fc1 = nn.Linear(42,hidden_dims[-1])
        self.fc1 = nn.Linear(1750,hidden_dims[-1])
        # self.fc2 = nn.Linear(hidden_dims[-1],hidden_dims[-1])
        self.output_layer = nn.Linear(hidden_dims[-1],output_dim)


    def forward(self,state):
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32)#,device=self.device)
        if(len(x.shape) == 1):
            x = x.unsqueeze(0)
        x = x.unsqueeze(1) # not sure if this is right?
        x = self.activation_fc(self.c1(x))
        x = self.activation_fc(self.c2(x))
        x = self.activation_fc(self.c3(x))
        x = torch.flatten(x,1) # need ot understand if this is correct - should be (N,something)
        # x = x.unsqueeze(0) # weird hack
        x = self.activation_fc(self.fc1(x))
        # x = self.activation_fc(self.fc2(x))
        x = self.output_layer(x)
        return x


class Trnsfrmr1(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dims=(20,20),activation_fc=F.relu,hiddenlayers=1):
        super(Trnsfrmr1,self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim,hidden_dims[0])
        self.positional_encoding = PositionalEncoding(d_model=10)
        self.embedding = nn.Embedding(4,10)
        self.hiddenlayers = hiddenlayers
        # self.hidden_layers = nn.ModuleList()
        # self.t1 = nn.Transformer(input_dim,input_dim)
        self.e1 = nn.TransformerEncoderLayer(input_dim,input_dim,dim_feedforward=20)#,activation=F.tanh)
        self.e2 = nn.TransformerEncoderLayer(input_dim,input_dim,dim_feedforward=40)
        self.e3 = nn.TransformerEncoderLayer(input_dim,input_dim,dim_feedforward=80)
        # self.e3 = nn.TransformerEncoderLayer(hidden_dims[0],hidden_dims[0])
        # self.d1 = nn.TransformerDecoderLayer(hidden_dims[0],hidden_dims[0])
        # self.e2 = nn.TransformerEncoder(10,10)
        # self.d2 = nn.TransformerDecoder(10,10)
        # self.fc1 = nn.Linear(6,13) # need to work on this layer
        self.fc0 = nn.Linear(input_dim,input_dim)
        self.fc1 = nn.Linear(input_dim,hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1],input_dim)
        self.fc4 = nn.Linear(hidden_dims[0],input_dim)
        self.output_layer = nn.Linear(input_dim,output_dim)


    def forward(self,state,mask=None):
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32)#,device=self.device)
        if(len(x.shape) == 1):
            x = x.unsqueeze(0)
        # x = x.unsqueeze(1) # not sure if this is right?
        # x = self.positional_encoding(x)
        # src_mask = nn.Transformer.generate_square_subsequent_mask(len(x))
        # x = self.t1(x,src_mask)
        x = self.e1(x)
        # x = self.activation_fc(self.fc3(x))
        x = self.e2(x)
        # x = self.activation_fc(self.fc2(x))
        x = self.e3(x)
        x = torch.flatten(x,1) # need to understand if this is correct - should be (N,something)
        # x = x.unsqueeze(0) # weird hack
        if self.hiddenlayers == 2:
            x = self.activation_fc(self.fc1(x))
            x = self.activation_fc(self.fc4(x))
        elif self.hiddenlayers == 3:
            x = self.activation_fc(self.fc1(x))
            x = self.activation_fc(self.fc2(x))
            x = self.activation_fc(self.fc3(x))
        else:
            x = self.activation_fc(self.fc0(x))
        x = self.output_layer(x)
        return x


class Trnsfrmr2(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dims = [6,6],feedforward_dim = 9,activation_fc=F.relu):
        super(Trnsfrmr2,self).__init__()
        self.activation_fc = activation_fc
        self.transformer = nn.Transformer(input_dim,input_dim,hidden_dims[0],hidden_dims[1],feedforward_dim,activation=activation_fc)
        self.outputer = nn.Linear(input_dim,output_dim)


    def forward(self,state,mask=None):
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32)#,device=self.device)
        if(len(x.shape) == 1):
            x = x.unsqueeze(0)
        x = self.transformer(x)
        x = self.outputer(x)
        return x


class Dense91(nn.Module):
    def __init__(self,input_dim=9,output_dim=1,hidden_dims=(20,20),activation_fc=F.relu):
        super(Dense91,self).__init__()
        if len(hidden_dims) != 2:
            print('Poor hidden dims input, reverting to default')
            hidden_dims = (20,20)
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim,hidden_dims[0])
        self.positional_encoding = PositionalEncoding(d_model=10)
        # self.embedding = nn.Embedding(4,10)
        # self.hidden_layers = nn.ModuleList()
        self.fc0 = nn.Linear(hidden_dims[0],hidden_dims[1])
        # self.fc2 = nn.Linear(13,13)
        self.output_layer = nn.Linear(hidden_dims[1],output_dim) #,activation_fc=F.sigmoid)


    def forward(self,state,mask=None):
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32)#,device=self.device)
        if(len(x.shape) == 1):
            x = x.unsqueeze(0)
        # x = x.unsqueeze(1) # not sure if this is right?
        # x = self.positional_encoding(x)
        x = torch.flatten(x,1) # need ot understand if this is correct - should be (N,something)
        # x = x.unsqueeze(0) # weird hack
        x = self.activation_fc(self.input_layer(x))
        # x = self.activation_fc(self.fc2(x))
        x = self.activation_fc(self.fc0(x))
        x = self.output_layer(x)
        return x


class Dense2Layer(nn.Module):
    def __init__(self,input_dim=9,output_dim=1,hidden_dims=(20,20,20),activation_fc=F.relu):
        super(Dense91,self).__init__()
        if len(hidden_dims) != 3:
            print('Poor hidden dims input, reverting to default')
            hidden_dims = (20,20,20)
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim,hidden_dims[0])
        self.positional_encoding = PositionalEncoding(d_model=10)
        # self.embedding = nn.Embedding(4,10)
        # self.hidden_layers = nn.ModuleList()
        self.fc0 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc1 = nn.Linear(hidden_dims[1],hidden_dims[2])
        # self.fc2 = nn.Linear(13,13)
        self.output_layer = nn.Linear(hidden_dims[2],output_dim) #,activation_fc=F.sigmoid)


    def forward(self,state,mask=None):
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32)#,device=self.device)
        if(len(x.shape) == 1):
            x = x.unsqueeze(0)
        # x = x.unsqueeze(1) # not sure if this is right?
        # x = self.positional_encoding(x)
        x = torch.flatten(x,1) # need ot understand if this is correct - should be (N,something)
        # x = x.unsqueeze(0) # weird hack
        x = self.activation_fc(self.input_layer(x))
        # x = self.activation_fc(self.fc2(x))
        x = self.activation_fc(self.fc0(x))
        x = self.activation_fc(self.fc1(x))
        x = self.output_layer(x)
        return x


class Dense3Layer(nn.Module):
    def __init__(self,input_dim=9,output_dim=1,hidden_dims=(20,20,20,20),activation_fc=F.relu):
        super(Dense91,self).__init__()
        if len(hidden_dims) != 4:
            print('Poor hidden dims input, reverting to default')
            hidden_dims = (20,20,20,20)
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim,hidden_dims[0])
        self.positional_encoding = PositionalEncoding(d_model=10)
        # self.embedding = nn.Embedding(4,10)
        # self.hidden_layers = nn.ModuleList()
        self.fc0 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.fc1 = nn.Linear(hidden_dims[1],hidden_dims[2])
        self.fc2 = nn.Linear(hidden_dims[2],hidden_dims[3])
        # self.fc2 = nn.Linear(13,13)
        self.output_layer = nn.Linear(hidden_dims[3],output_dim) #,activation_fc=F.sigmoid)


    def forward(self,state,mask=None):
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32)#,device=self.device)
        if(len(x.shape) == 1):
            x = x.unsqueeze(0)
        # x = x.unsqueeze(1) # not sure if this is right?
        # x = self.positional_encoding(x)
        x = torch.flatten(x,1) # need ot understand if this is correct - should be (N,something)
        # x = x.unsqueeze(0) # weird hack
        x = self.activation_fc(self.input_layer(x))
        # x = self.activation_fc(self.fc2(x))
        x = self.activation_fc(self.fc0(x))
        x = self.activation_fc(self.fc1(x))
        x = self.activation_fc(self.fc2(x))
        x = self.output_layer(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class transformer(nn.Module): #https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, ntoken: int, nhead: int = 2, d_hid: int = 20,
                 nlayers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()


    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)


    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        if not isinstance(src,torch.Tensor):
            src = torch.tensor(src,dtype=torch.float32)#,device=self.device)
            src = src.unsqueeze(0)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)) #.to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


class RNN1(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dims=(15,9),activation_fc=F.relu):
        super(RNN1,self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim,hidden_dims[0])
        # self.hidden_layers = nn.ModuleList()
        self.positional_encoding = PositionalEncoding(d_model=10)
        self.embedding = nn.Embedding(4,10)
        self.r0 = nn.RNN(input_dim,hidden_dims[0],3)
        self.r1 = nn.RNN(hidden_dims[0],hidden_dims[0],3)#,nonlinearity='relu')
        self.r2 = nn.RNN(hidden_dims[0],hidden_dims[0],3)#,nonlinearity='relu')
        self.r3 = nn.RNN(hidden_dims[0],hidden_dims[0],3)#,nonlinearity='relu')
        self.fc1 = nn.Linear(hidden_dims[0],hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0],hidden_dims[-1])
        self.output_layer = nn.Linear(hidden_dims[-1],output_dim)


    def forward(self,state):
        x = state
        # if not isinstance(x,torch.Tensor):
        #     x = torch.tensor(x,dtype=torch.float32)#,device=self.device)
        #     x = x.unsqueeze(0)
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32)#,device=self.device)
        if(len(x.shape) == 1):
            x = x.unsqueeze(0)
        # if x.shape[0] > 1:
        #     x = x.unsqueeze(1)
        # x = self.activation_fc(self.r1(x))
        # h0 = torch.autograd.Variable(torch.zeros(10,x.size(0),10))
        # hn = x*0
        x, hn = self.r0(x)#,hn)#,h0)
        x,hn = self.r1(x,hn)
        x,hn = self.r2(x,hn)
        x,hn = self.r3(x,hn)
        # x,hn = self.r2(x,hn)
        # x,hn = self.r1(x)#,hn)#,h0)
        # x = self.activation_fc(self.r2(x))
        x = torch.flatten(x,1) # might not need this?
        # x = self.activation_fc(self.fc1(x))
        x = self.activation_fc(self.fc2(x))
        x = self.output_layer(x)
        if x.shape[1] == 1:
            x.flatten()
        return x


# class PINN(): # need to understand how to create a physics-informed neural network from this
#     """
#     Based on code by Maziar Raissi, Paris Perdikaris, and George Em Karniadakis found at: https://github.com/jayroxis/PINNs/blob/master/Burgers%20Equation/Burgers%20Inference%20(PyTorch).ipynb

#     @article{raissi2017physicsI, title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations}, author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em}, journal={arXiv preprint arXiv:1711.10561}, year={2017} }

#     @article{raissi2017physicsII, title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations}, author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em}, journal={arXiv preprint arXiv:1711.10566}, year={2017} }

#     """
#     def __init__(self,X,U,T):
#         self.T = torch.tensor(T) # might not need this
#         self.X = torch.tensor(X[:, 0:1], requires_grad=True) # variable for measured angle, or state?
#         self.U = torch.tensor(U[:, 0:1], requires_grad=True)
#         self.dnn = Dense91()#.to(device=)
#         self.optimizer = torch.optim.LBFGS(
#             self.dnn.parameters(), 
#             lr=1.0, 
#             max_iter=50000, 
#             max_eval=50000, 
#             history_size=50,
#             tolerance_grad=1e-5, 
#             tolerance_change=1.0 * np.finfo(float).eps,
#             line_search_fn="strong_wolfe"       # can be "strong_wolfe"
#         )
#         self.iter = 0


#     def net_u(self, x, t): # prediction (used for prediction loss)
#         u = self.dnn(torch.cat([x, t], dim=1))
#         return u


#     def net_f(self, x, t): # physics informed loss
#         """ The pytorch autograd version of calculating residual """
#         u = self.net_u(x, t)

#         u_t = torch.autograd.grad(
#             u, t, 
#             grad_outputs=torch.ones_like(u),
#             retain_graph=True,
#             create_graph=True
#         )[0]
#         u_x = torch.autograd.grad(
#             u, x, 
#             grad_outputs=torch.ones_like(u),
#             retain_graph=True,
#             create_graph=True
#         )[0]
#         u_xx = torch.autograd.grad(
#             u_x, x, 
#             grad_outputs=torch.ones_like(u_x),
#             retain_graph=True,
#             create_graph=True
#         )[0]

#         f = u_t + u * u_x - self.nu * u_xx # replace with the correct equation
#         return f


#     def loss_func(self):
#         self.optimizer.zero_grad()
        
#         # update loss functions to be appropriate equations
#         u_pred = self.net_u(self.x_u, self.t_u)
#         f_pred = self.net_f(self.x_f, self.t_f)
#         loss_u = torch.mean((self.u - u_pred) ** 2)
#         loss_f = torch.mean(f_pred ** 2)
        
#         loss = loss_u + loss_f
        
#         loss.backward()
#         self.iter += 1
#         if self.iter % 100 == 0:
#             print(
#                 'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
#             )
#         return loss
    
    
#     def train(self):
#         self.dnn.train()

#         # Backward and optimize
#         self.optimizer.step(self.loss_func)


#     def predict(self, X):
#         x = torch.tensor(X[:, 0:1], requires_grad=True).float()#.to(device)
#         t = torch.tensor(X[:, 1:2], requires_grad=True).float()#.to(device)

#         self.dnn.eval()
#         u = self.net_u(x, t)
#         f = self.net_f(x, t)
#         u = u.detach()#.cpu().numpy()
#         f = f.detach()#.cpu().numpy()
#         return u, f

class angleCalc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles, state): # angles - siderealtime, ra, dec, dra, ddec?; state - Jtime, x, y, z, dx, dy, dz? # output - range? Sidereal time instead of Jtime?
        ctx.save_for_backward(input)
        # To be finished - forward direction range measurements given angles and orbit shape
        # mu = 398600.5
        # I = [cosD*cosA,cosD*sinA,sinD]
        # S = s*[cosL*cosS,cosL*sinS,sinL]
        # r = R*I + S
        # rd = Rd*I + R*Id + Sd
        # rdd = Rdd*I + 2*Rd*Id + R*Idd + sdd
        # rdd = -mu * r / abs(r**3)
        # find R
        range = 0
        return range


    @staticmethod
    def backward(ctx, range):
        input, = ctx.saved_tensors
        angles = [0, 0, 0, 0]
        # To be finished - forward direction angles from range
        return angles, None


class aorbitCalc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles): # angles - ra, dec, dra, ddec?; state - Jtime, x, y, z, dx, dy, dz? # output - range?
        ctx.save_for_backward(input)
        # To be finished - forward direction range measurements given angles and orbit shape
        orbit = [0, 0, 0, 0, 0, 0, 0] # first variable is time?
        return orbit


    @staticmethod
    def backward(ctx, orbit):
        input, = ctx.saved_tensors
        angles = [0, 0, 0] # first variable is time?
        # To be finished - forward direction angles from range
        return angles


class orbitPINN(orbitFunctions.orbitFunctions):
    """
    Based on code by Maziar Raissi, Paris Perdikaris, and George Em Karniadakis found at: https://github.com/jayroxis/PINNs/blob/master/Burgers%20Equation/Burgers%20Inference%20(PyTorch).ipynb

    @article{raissi2017physicsI, title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations}, author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em}, journal={arXiv preprint arXiv:1711.10561}, year={2017} }

    @article{raissi2017physicsII, title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations}, author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em}, journal={arXiv preprint arXiv:1711.10566}, year={2017} }


    Example from Medium.com:
    class Net(nn.Module):
    def __init__(self, *args):
        ...
        # make r a differentiable parameter included in self.parameters()
        self.r = nn.Parameter(data=torch.tensor([0.]))
        ...

    def physics_loss_discovery(model: torch.nn.Module):
        ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True).to(DEVICE)
        temps = model(ts)
        dT = grad(temps, ts)[0]
        # use the differentiable parameter instead
        pde = model.r * (Tenv - temps) - dT
        
        return torch.mean(pde**2)

    """
    def __init__(self):
        self.x = np.random.rand(3)*1e6
        self.A = self.createOrbitA6(self.x)
        # self.dnn = RNN1(input_dim=5,output_dim=1,hidden_dims=(8,8),activation_fc=F.sigmoid)#.to(device=)
        # self.dnn = Dense91(input_dim=10,output_dim=1,hidden_dims=(100,100),activation_fc=F.relu)
        self.dnn = Trnsfrmr1(input_dim=10,output_dim=1,hidden_dims=(20,20),activation_fc=F.relu)
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.x_u = torch.tensor([])
        self.t_u = torch.tensor([])
        self.x_f = torch.tensor([])
        self.t_f = torch.tensor([])
        self.u = None
        self.truthdata = torch.tensor([])
        self.rangeConversion = 72000
        # self.otherFuncs = MLdataManipulation()
        # self.optimizer = torch.optim.LBFGS(
        #     self.dnn.parameters(), 
        #     lr=1.0, 
        #     max_iter=50000, 
        #     max_eval=50000, 
        #     history_size=50,
        #     tolerance_grad=1e-5, 
        #     tolerance_change=1.0 * np.finfo(float).eps,
        #     line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        # )
        self.optimizer = torch.optim.Adam(
            self.dnn.parameters(), 
            lr=1.0, 
        )
        self.iter = 0
        self.OF = orbitFunctions.orbitFunctions()
        self.conversion = 1e6
        torch.autograd.set_detect_anomaly(True)


    def saveNN(self,filename = 'models/pinn_weights'):
        weights = copy.deepcopy(self.dnn.state_dict())
        torch.save(weights,filename)


    def organizeData(self, X, T):
        d2r = torch.pi/180
        Dtrain = torch.stack([torch.sin(T[0] * d2r), torch.cos(T[0] * d2r), torch.sin(X[0] * d2r), torch.cos(X[0] * d2r), torch.sin(X[1] * d2r), torch.cos(X[1] * d2r), torch.sin(X[2] * d2r), torch.cos(X[2] * d2r), torch.sin(X[3] * d2r), torch.cos(X[3] * d2r)]) # normalize data
        return Dtrain.transpose(-1,0)


    def net_u(self, x, t): # prediction (used for prediction loss)
        input = self.organizeData(x,t)
        if input.shape[0] == 5:
            input = torch.transpose(input,0,1)
        u = self.dnn(input)#, dim=1))
        # u = self.dnn(torch.cat([x, t]),0,1)#, dim=1))
        return u


    def net_f(self, x, t): # physics informed loss
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, t)
        # torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),allow_unused=True)
        t.requires_grad = True
        x.requires_grad = True
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            allow_unused=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            allow_unused=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            allow_unused=True,
            create_graph=True
        )[0]

        # f = u_t + u * u_x - self.nu * u_xx # physics-involved error calculation (physics propagated elsewhere?)
        f = -398600.5 / np.sqrt(np.sum(x**2)) * u_xx # orbit model?
        """
        would like to go from angular measurements to range (i.e. angles and angle rates) or how to get angular measurements out with the physics involved
        PINN seems to rely on derivative - how is this usable for angle measurements informing range prediction?  How would this be coded?
        PINN assumes partially differentiable equations (PDE or ODE), so need an equation relating range to angles for an orbital object?
        Inputs  - likely angular measurements and angular rates
        Outputs - range estimation (like Gauss's method or KF)
        """
        # a = torch.autograd.grad(inputs=orbitFunctions.orbitFunctions.AzEl2HDec())
        # f = self.OF.AzEl2HDec(u_x[0],u_x[1],self.OF.Sensor)
        f = self.OF.orbitSensorModelAOrate(u_x,[u_t]) # need to modify this to better
        return f


    def range_err(self, x, t):
        # input angles, output range?
        # output = dR w/ respect to angles
        # dRdA = drange dazimuth (or drightascension)
        # dRdE = drange delevation (or ddeclination)
        u = self.net_u(x, t) * self.rangeConversion
        mu = 398600.5
        Rearth = 6378.0
        stim2seconds = (2*torch.pi/86400)
        u = torch.reshape(u,[u.shape[1],u.shape[0]])
        u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),  create_graph=True)[0]# * stim2seconds
        u_tt = torch.autograd.grad(u_t,t,grad_outputs=torch.ones_like(u),  create_graph=True)[0]# * stim2seconds
        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),  create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u_x),  create_graph=True)[0]
        A = x[0] #torch.linspace(-np.pi,np.pi,steps=t.shape[1]).view(-1,1)#.requires_grad(True) # or could try 100 steps?
        D = x[1] #torch.linspace(0,np.pi,steps=t.shape[1]).view(-1,1)#.requires_grad(True)
        dA = x[2] #torch.linspace(-np.pi/10,np.pi/10,steps=t.shape[1]).view(-1,1)#.requires_grad(True)
        dD = x[3] #torch.linspace(-np.pi/10,np.pi/10,steps=t.shape[1]).view(-1,1)#.requires_grad(True)
        # dt = torch.diff(t)
        # ddt = torch.diff(dt)
        # tf = torch.linspace(0,2*np.pi,steps=t.shape[1]).view(-1,1)
        Ix = torch.cos(D)*torch.cos(A)
        Iy = torch.cos(D)*torch.sin(A)
        Iz = torch.sin(D)
        I = torch.stack([Ix.flatten(),Iy.flatten(),Iz.flatten()])
        Idx = -torch.cos(D)*torch.sin(A)*dA - torch.sin(D)*torch.cos(A)*dD
        Idy = torch.cos(D)*torch.cos(A)*dA - torch.sin(D)*torch.sin(A)*dD
        Idz = torch.cos(D)*dD
        Id = torch.stack([Idx.flatten(),Idy.flatten(),Idz.flatten()])
        ddA = 0 # not really, but it's so small it is difficult to measure in the timespan given
        ddD = 0 # not really, but it's so small - you get the point
        Iddx = -torch.sin(D)*torch.sin(A)*dD*dA + torch.cos(D)*torch.cos(A)*dA - torch.cos(D)*torch.sin(A)*ddA + torch.sin(D)*torch.sin(A)*dD*dA + torch.cos(D)*torch.cos(A)*dD - torch.sin(D)*torch.cos(A)*ddD
        Iddy = (-torch.sin(D)*torch.cos(A)*dD*dA - torch.cos(D)*torch.sin(A)*dA + torch.cos(D)*torch.cos(A)*ddA) - (torch.sin(D)*torch.cos(A)*dD*dA + torch.cos(D)*torch.sin(A)*dD + torch.sin(D)*torch.sin(A)*ddD)
        Iddz = torch.cos(D)*ddD - torch.sin(D)*dD
        Idd = torch.stack([Iddx.flatten(),Iddy.flatten(),Iddz.flatten()])
        # If using t as sidereal time
        s = Rearth * torch.cat([np.cos(self.OF.Sensor.latitude_rad)*torch.cos(t),np.cos(self.OF.Sensor.latitude_rad)*torch.sin(t),torch.ones_like(t) * np.sin(self.OF.Sensor.latitude_rad)])
        sd = Rearth * torch.cat([-np.cos(self.OF.Sensor.latitude_rad)*torch.sin(t)*stim2seconds,np.cos(self.OF.Sensor.latitude_rad)*torch.cos(t)*stim2seconds, 0. * torch.ones_like(t)])# * stim2seconds # time is in radians, need time in seconds
        sdd = Rearth * torch.cat([-np.cos(self.OF.Sensor.latitude_rad)*torch.cos(t)*stim2seconds, -np.cos(self.OF.Sensor.latitude_rad)*torch.sin(t)*stim2seconds, 0. * torch.ones_like(t)])# * stim2seconds
        r = u * I + s
        rd = u_t * I + u * Id + sd
        rdd1 = u_tt * I + 2 * u_t * Id + u * Idd + sdd
        rdd2 = -mu * r / torch.linalg.norm(r,2,0)**3
        rddloss = rdd1 - rdd2
        return rddloss / self.rangeConversion


    def orbitErr(self,x,t,dt=60):
        sens = self.OF.orbitSensorModelAOrate(x,[t])
        return sens


    def PDE(self,xh,dt):
        # dt = self.dt
        r = np.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -398600.5/r**3
        A1 = [0,0,0,1,0,0]
        A2 = [0,0,0,0,1,0]
        A3 = [0,0,0,0,0,1]
        A4 = [cr,0,0,0,0,0]
        A5 = [0,cr,0,0,0,0]
        A6 = [0,0,cr,0,0,0]
        A = np.array([A1,A2,A3,A4,A5,A6])
        xh1 = np.matmul(A,xh)*dt + xh #+ xw


    def lossPDE(self,x_PDE):
        g = x_PDE.clone()
        g.requires_grad = True
        f = self.dnn.forward(g)
        f_x = torch.autograd.grad(f,g,torch.ones([x_PDE.shape[0],1]),retain_graph=True,create_graph=True)[0]
        lossPDE = self.loss_function(f_x,self.PDE(g)) # not sure how PDE is defined
        return lossPDE


    def loss_func(self):
        # if zgrad:
        self.optimizer.zero_grad()
        
        # update loss functions to be appropriate equations
        u_pred = self.net_u(self.x_u, self.t_u) #TBD - how to set self.x_u and self.t_u - are these inputs that change every time?
        # f_pred = self.net_f(self.x_f, self.t_f)
        # if self.u == None:
        self.u = torch.mean(u_pred) # when does self.u need to be updated?
        f_pred = self.range_err(self.x_f, self.t_f)
        if len(u_pred.shape) > 1 and u_pred.shape[1] == 1:
            u_pred = u_pred.flatten()
        loss_u = torch.mean((u_pred - self.truthdata) ** 2) # where does self.u come from?
        loss_f = torch.linalg.norm(f_pred,1) # 1 or 2 norm?
        # loss_f = torch.mean(f_pred ** 2)
        
        loss = loss_u + loss_f
        
        loss.backward(retain_graph=True)
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
        return loss


    def train(self,X, T, Truth):
        T = np.array([T])
        self.x_u = torch.tensor(X.copy(),requires_grad=True).float() #torch.tensor(X[:,0:4],requires_grad=True).float() #.to_device()
        self.t_u = torch.tensor(T.copy(),requires_grad=True).float() #torch.tensor(X[:,4],requires_grad=True).float()
        self.x_f = torch.tensor(X.copy(),requires_grad=True).float() #torch.tensor(X[:,0:4],requires_grad=True).float() #.to_device()
        self.t_f = torch.tensor(T.copy(),requires_grad=True).float() #torch.tensor(X[:,4],requires_grad=True).float()
        self.truthdata = torch.tensor(Truth).float() / self.rangeConversion
        if self.dnn.training == False:
            self.dnn.train()

        # Backward and optimize
        loss = self.loss_func()
        # loss.backward(retain_graph=True)
        self.optimizer.step()
        # self.optimizer.step(self.loss_func) # is this sufficient or do I need the 3 lines above?


    def predict(self, X, T, denormalize=1):
        if self.dnn.training == True:
            self.dnn.eval()
        T = np.array([T])
        self.x_u = torch.tensor(X.copy(),requires_grad=True).float() #torch.tensor(X[:,0:4],requires_grad=True).float() #.to_device()
        self.t_u = torch.tensor(T.copy(),requires_grad=True).float() #torch.tensor(X[:,4],requires_grad=True).float()
        u = self.net_u(self.x_u, self.t_u)
        if denormalize:
            u = u * self.rangeConversion
        return u


    def organize_data(self,data):
        dataNew = []
        for ii in range(0,len(data)):
            dataNew.append(data[ii][0:-1])
        return np.transpose(dataNew)


class MLdataManipulation():
    
    
    def rangetranslation(self,range,direction=1,mag=72000):
        output = range
        if direction == 0: # forward
            output = range / mag
        elif direction == 1: # backward
            output = range * mag
        return output

    
    def organizeDataInput(self,observer,siderealtime,dataseting=0,rangenorm=72000):
        d2r = np.pi/180
        if dataseting == 2:
            Dtrain = np.transpose([siderealtime.astype(np.float64) / (2*np.pi), np.sin(observer[1].astype(np.float64) * d2r), np.cos(observer[1].astype(np.float64) * d2r), np.sin(observer[2].astype(np.float64) * d2r), np.cos(observer[2].astype(np.float64) * d2r), np.sin(observer[8].astype(np.float64) * d2r), np.cos(observer[8].astype(np.float64) * d2r), np.sin(observer[9].astype(np.float64) * d2r), np.cos(observer[9].astype(np.float64) * d2r)]) # normalize data
            Rtrain = np.array(observer[3] / rangenorm ) # normalize range
        elif dataseting == 3:
            Dtrain = np.transpose([siderealtime[0:-1].astype(np.float64) / (2*np.pi), np.sin(observer[1][0:-1].astype(np.float64) * d2r), np.cos(observer[1][0:-1].astype(np.float64) * d2r), np.sin(observer[2][0:-1].astype(np.float64) * d2r), np.cos(observer[2][0:-1].astype(np.float64) * d2r), np.sin(observer[8][0:-1].astype(np.float64) * d2r), np.cos(observer[8][0:-1].astype(np.float64) * d2r), np.sin(observer[9][0:-1].astype(np.float64) * d2r), np.cos(observer[9][0:-1].astype(np.float64) * d2r),
                    siderealtime[1:].astype(np.float64) / (2*np.pi), np.sin(observer[1][1:].astype(np.float64) * d2r), np.cos(observer[1][1:].astype(np.float64) * d2r), np.sin(observer[2][1:].astype(np.float64) * d2r), np.cos(observer[2][1:].astype(np.float64) * d2r), np.sin(observer[8][1:].astype(np.float64) * d2r), np.cos(observer[8][1:].astype(np.float64) * d2r), np.sin(observer[9][1:].astype(np.float64) * d2r), np.cos(observer[9][1:].astype(np.float64) * d2r)]) # normalize data
            Rtrain = np.array(observer[3] / rangenorm ) # normalize range
        elif dataseting == 4:
            Dtrain = np.transpose([np.sin(siderealtime.astype(np.float64) * d2r),np.cos(siderealtime.astype(np.float64) * d2r), np.sin(observer[1].astype(np.float64) * d2r), np.cos(observer[1].astype(np.float64) * d2r), np.sin(observer[2].astype(np.float64) * d2r), np.cos(observer[2].astype(np.float64) * d2r), np.sin(observer[8].astype(np.float64) * d2r), np.cos(observer[8].astype(np.float64) * d2r), np.sin(observer[9].astype(np.float64) * d2r), np.cos(observer[9].astype(np.float64) * d2r)]) # normalize data
            Rtrain = np.array(observer[3] / rangenorm ) # normalize range
        elif dataseting == 5:
            Dtrain = np.transpose([np.sin(siderealtime[0:-1].astype(np.float64) * d2r),np.cos(siderealtime[0:-1].astype(np.float64) * d2r), np.sin(observer[1][0:-1].astype(np.float64) * d2r), np.cos(observer[1][0:-1].astype(np.float64) * d2r), np.sin(observer[2][0:-1].astype(np.float64) * d2r), np.cos(observer[2][0:-1].astype(np.float64) * d2r), np.sin(observer[8][0:-1].astype(np.float64) * d2r), np.cos(observer[8][0:-1].astype(np.float64) * d2r), np.sin(observer[9][0:-1].astype(np.float64) * d2r), np.cos(observer[9][0:-1].astype(np.float64) * d2r),
                                   np.sin(siderealtime[1:].astype(np.float64) * d2r),np.cos(siderealtime[1:].astype(np.float64) * d2r), np.sin(observer[1][1:].astype(np.float64) * d2r), np.cos(observer[1][1:].astype(np.float64) * d2r), np.sin(observer[2][1:].astype(np.float64) * d2r), np.cos(observer[2][1:].astype(np.float64) * d2r), np.sin(observer[8][1:].astype(np.float64) * d2r), np.cos(observer[8][1:].astype(np.float64) * d2r), np.sin(observer[9][1:].astype(np.float64) * d2r), np.cos(observer[9][1:].astype(np.float64) * d2r)]) # normalize data
            Rtrain = np.array(observer[3] / rangenorm ) # normalize range
        elif dataseting == 1:
            Dtrain = np.transpose([siderealtime[0:-1].astype(np.float64) / (2*np.pi),observer[1][0:-1].astype(np.float64) / 360,observer[2][0:-1].astype(np.float64) /360,observer[8][0:-1].astype(np.float64) / 360,observer[9][0:-1].astype(np.float64) / 360,siderealtime[1:].astype(np.float64) / (2*np.pi),observer[1][1:].astype(np.float64) / 360,observer[2][1:].astype(np.float64) /360,observer[8][1:].astype(np.float64) / 360,observer[9][1:].astype(np.float64) / 360]) # normalize data
            Rtrain = np.array(observer[3][0:-1].astype(np.float64) / rangenorm ) # normalize range
        else:
            Dtrain = np.transpose([siderealtime.astype(np.float64) / (2*np.pi),observer[1].astype(np.float64) / 360,observer[2].astype(np.float64) /360,observer[8].astype(np.float64) / 360,observer[9].astype(np.float64) / 360]) # normalize data
            Rtrain = np.array(observer[3].astype(np.float64) / rangenorm ) # normalize range
        return Dtrain, Rtrain


    def grabInputNoNorm(self,observer,siderealtime,dataseting=0):
        d2r = np.pi/180
        if dataseting == 2:
            Dtrain = np.transpose([siderealtime.astype(np.float64),observer[1].astype(np.float64),observer[1].astype(np.float64),observer[2].astype(np.float64),observer[2].astype(np.float64),observer[8].astype(np.float64),observer[8].astype(np.float64),observer[9].astype(np.float64),observer[9].astype(np.float64)]) # normalize data
            Rtrain = np.array(observer[3])
        elif dataseting == 3:
            Dtrain = np.transpose([siderealtime[0:-1].astype(np.float64), observer[1][0:-1].astype(np.float64), observer[1][0:-1].astype(np.float64), observer[2][0:-1].astype(np.float64), observer[2][0:-1].astype(np.float64), observer[8][0:-1].astype(np.float64), observer[8][0:-1].astype(np.float64),observer[9][0:-1].astype(np.float64),observer[9][0:-1].astype(np.float64),
                    siderealtime[1:].astype(np.float64), observer[1][1:].astype(np.float64),observer[1][1:].astype(np.float64),observer[2][1:].astype(np.float64), observer[2][1:].astype(np.float64) ,observer[8][1:].astype(np.float64) ,observer[8][1:].astype(np.float64),observer[9][1:].astype(np.float64),observer[9][1:].astype(np.float64)]) # normalize data
            Rtrain = np.array(observer[3])
        elif dataseting == 1:
            Dtrain = np.transpose([siderealtime[0:-1].astype(np.float64),observer[1][0:-1].astype(np.float64),observer[2][0:-1].astype(np.float64) ,observer[8][0:-1].astype(np.float64),observer[9][0:-1].astype(np.float64),siderealtime[1:].astype(np.float64),observer[1][1:].astype(np.float64),observer[2][1:].astype(np.float64),observer[8][1:].astype(np.float64),observer[9][1:].astype(np.float64)]) # normalize data
            Rtrain = np.array(observer[3][0:-1].astype(np.float64)) # normalize range
        elif dataseting == 5:
            Dtrain = np.transpose([siderealtime[0:-1].astype(np.float64),observer[1][0:-1].astype(np.float64),observer[2][0:-1].astype(np.float64),observer[8][0:-1].astype(np.float64),observer[9][0:-1].astype(np.float64),
                                   siderealtime[1:].astype(np.float64),observer[1][1:].astype(np.float64),observer[2][1:].astype(np.float64),observer[8][1:].astype(np.float64),observer[9][1:].astype(np.float64)]) # normalize data
            Rtrain = np.array(observer[3].astype(np.float64)) # normalize range
        else:
            Dtrain = np.transpose([siderealtime.astype(np.float64),observer[1].astype(np.float64),observer[2].astype(np.float64),observer[8].astype(np.float64),observer[9].astype(np.float64)]) # normalize data
            Rtrain = np.array(observer[3].astype(np.float64)) # normalize range
        return Dtrain, Rtrain

