# TO DO:
# Instead of using bb separate nets, append in agents inputs so that shapes are (B,T*3+2) and (B,M,T*3+2)
import torch
from torch.nn import Linear, ReLU, Sequential,Dropout

# Model for updating node attritbutes
class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, hid_channels):
        super().__init__()

        layers = []
        for l in range(n_layers):
            layers.append(Dropout(0.5))
            layers.append(Linear(in_channels, hid_channels))
            layers.append(ReLU())
            in_channels = hid_channels
            hid_channels *= 2
            if l==n_layers-2:   hid_channels = out_channels
                  
        # if self.norm:  layers.append(LayerNorm(node_out))

        self.mlp = Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        # if self.residuals:
        #     out = out + x
        return out



class AgentsModule(torch.nn.Module):
    def __init__(self, n_out_ag = 1024):
        super().__init__()

        n_channels = 10
        n_coords = 3
        n_in = n_channels*n_coords+2
        n_layers = 5
        hid_channels = 128

        self.mlp_states_ego = MLP(n_in, n_out_ag, n_layers, hid_channels)
        self.mlp_states_agents = MLP(n_in, n_out_ag, n_layers, hid_channels)

    # x_ego: (B,T,3)
    # bb_ego: (B,2)
    # x_agents: (B,M,T,3)
    # bb_agents: (B,M,2)
    # out: (B,2*n_out_ag)
    def forward(self, x_ego, bb_ego, x_agents, bb_agents):

        agsshape = x_agents.shape
        x_ego = x_ego.view(agsshape[0],-1)
        x_ego = torch.cat([x_ego,bb_ego],dim=-1)
        x_agents = x_agents.view(agsshape[0],agsshape[1],-1)
        x_agents = torch.cat([x_agents,bb_agents],dim=-1)
        
        x_ego = self.mlp_states_ego(x_ego)
        x_agents = self.mlp_states_agents(x_agents)

        # Sum all agents activations in the agents channels per each batch
        mat = torch.ones(agsshape[0],1,agsshape[1],device=x_agents.device)
        x_agents = torch.bmm(mat,x_agents).squeeze(1)

        out = torch.cat([x_ego, x_agents],dim=-1)

        return out
    

if __name__=="__main__":

    batchsize = 5
    maxag = 64
    timesize = 10

    x_ego = torch.randn(batchsize,timesize,3)
    bb_ego = torch.randn(batchsize,2)
    x_agents = torch.randn(batchsize,maxag,timesize,3)
    bb_agents = torch.randn(batchsize,maxag,2)

    agentsmodule = AgentsModule(n_out_ag = 1024)

    out = agentsmodule(x_ego, bb_ego, x_agents, bb_agents)
    print(out.shape)