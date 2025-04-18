import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, GraphConv,RGCNConv
from util import get_act_layer


class GNN(nn.Module):
    def __init__(self, args, node_feat_dim, edge_feat_dim):
        super(GNN, self).__init__()
        self.num_node_feat = node_feat_dim
        self.num_edge_feat = edge_feat_dim
        self.num_layers = args.num_layers
        self.num_hid = args.num_g_hid
        self.num_e_hid = args.num_e_hid
        self.num_out = args.gnn_out_dim
        self.model_type = args.gnn_type
        self.dropout = args.dropout
        self.convs = nn.ModuleList()
        self.act_type = args.act_type
        self.act_layer = get_act_layer(self.act_type)
        self.gnn_act_layer = get_act_layer(args.gnn_act_type)
        cov_layer = self.build_cov_layer(self.model_type)

        for l in range(self.num_layers):
            hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
            hidden_output_dim = self.num_out if l == self.num_layers - 1 else self.num_hid

            if self.model_type == "GIN" or self.model_type == "GAT" \
                    or self.model_type == "GCN":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))
            elif self.model_type == "RGCN":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim, self.num_edge_feat))
            else:
                raise NotImplementedError("Unsupported model type!")

    def build_cov_layer(self, model_type):
        if model_type == "GIN":
            return lambda in_ch, hid_ch : GINConv(nn= nn.Sequential(
                nn.Linear(in_ch, hid_ch), self.gnn_act_layer, nn.Linear(hid_ch, hid_ch)) )
        elif model_type == "RGCN":
            return lambda in_ch, hid_ch, num_edge_feat : RGCNConv(
                in_channels=in_ch, out_channels=hid_ch, num_relations=self.num_edge_feat)
        elif model_type == "GAT":
            return GATConv
        elif model_type == "GCN":
            return GraphConv
        else:
            raise NotImplementedError("Unsupported model type!")

    def forward(self, x, edge_index, edge_attr = None, att_bias = None):

        for i in range(self.num_layers):
            if self.model_type == "GIN" or self.model_type == "GAT" or self.model_type =="GCN":
                x = self.convs[i](x, edge_index)
            elif self.model_type == "RGCN":
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                print("Unsupported model type!")

            if i < self.num_layers - 1:
                x = self.act_layer(x)
                x = F.dropout(x, p = self.dropout, training=self.training)
        return x