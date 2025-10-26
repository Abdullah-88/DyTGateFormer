import torch
from torch import nn, Tensor
from typing import List
import math

class VectorDynamicTanh(nn.Module):
    def __init__(self, input_shape):
    
        super().__init__()
        
           
        self.alpha = nn.Parameter(torch.randn(input_shape))
   

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        
        return x







class DyTGate(nn.Module):
    def __init__(self, input_shape):
    
        super().__init__()
        
           
        self.proj = nn.Linear(input_shape,input_shape,bias=False)
        self.vdyt = VectorDynamicTanh(input_shape)
       

    def forward(self, x):
        u,v = x,x
        u = self.proj(x)
        u = self.vdyt(x)
        g = u * v
        
        return g


     
        

class LocalMappingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
           
        self.mapping = DyTGate(dim)
        self.norm = nn.LayerNorm(dim,elementwise_affine=False)
      
             	   
    def forward(self, x):
    
        x = self.norm(x) 
        x = self.mapping(x)   	
      
        return x
    	

class GlobalMappingUnit(nn.Module):
    
    def __init__(self, dim, num_heads):
            
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = dim
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim,elementwise_affine=False)
        assert self.head_dim * num_heads == self.hidden_dim 

       
        self.probe = DyTGate(self.hidden_dim)
        self.state = DyTGate(self.hidden_dim)
        self.readout = DyTGate(self.hidden_dim)

    def forward(self, x):
        
        batch_size, seq_len, _ = x.size()

        x = self.norm(x)
        P = self.probe(x)  
        S = self.state(x)
       

        
        P = P.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        SA = S.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        SR = S.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

       
        attention_scores = P @ SA.transpose(-1, -2) / math.sqrt(self.head_dim)

       

        
        attention_weights = torch.softmax(attention_scores, dim=-1)

        

       
        context = attention_weights @ SR

        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        
        return self.readout(context)         




class DyTGateBlock(nn.Module):
    def __init__(self, d_model,heads):
        super().__init__()
       
         
        self.local_mapping = LocalMappingUnit(d_model)
        self.global_mapping = GlobalMappingUnit(d_model,heads)
        
    
        
        
        
    def forward(self, x):
                  
        residual = x
        
        x = self.global_mapping(x)
    
        x = x + residual
        
        residual = x
        
        x = self.local_mapping(x)
        
                                          
        out = x + residual
        
        
        return out



class DytGateFormer(nn.Module):
    def __init__(self, d_model,heads, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[DyTGateBlock(d_model,heads) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








