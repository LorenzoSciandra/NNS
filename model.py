import torch
import torch.nn as nn
import torch.nn.functional as F
from nc.ddn_modules import ClosestETFGeometryLayer, FeaturesMovingAverageLayer
from models.resnet import resnet_set
from models.cnn import cnn_set
from models.mlp import mlp_set
from models.vgg import vgg_set
from dataset import DATASET_CLASSES, DATASET_FEATURES, DATASET_NUM_CHANNELS, DATASET_FLATTEN_FEATURES


def return_layer_list(model_type:str = "resnet10", dataset:str = 'cifar10', batch_norm: bool = False) -> list:
    
    if model_type.startswith('mlp'):
        return mlp_set(DATASET_CLASSES[dataset], DATASET_FEATURES[dataset], num_layers=int(model_type.split('_')[-1]))
    elif model_type == 'cnn':
        return cnn_set(DATASET_CLASSES[dataset], DATASET_NUM_CHANNELS[dataset], DATASET_FLATTEN_FEATURES[dataset])
    elif 'resnet' in model_type:
        return resnet_set(model_type, DATASET_CLASSES[dataset], DATASET_NUM_CHANNELS[dataset])
    elif 'vgg' in model_type:
        return vgg_set(model_type, DATASET_CLASSES[dataset], batch_norm, DATASET_NUM_CHANNELS[dataset])
        
    else:
        raise AttributeError("Set of layers not available")
    
    

class Extractor(nn.Module):
    def __init__(self, layers, num_classes):
        super(Extractor, self).__init__()
        self.num_layers = len(layers)
        self.net = nn.Sequential(*layers[:self.num_layers])
        self.init_weights()
        self.all_embs = {}
        self.all_flatten_dims = {}
        self.num_classes = num_classes
        self.save = False
        self.trimmed = False
        self.normalize = False
        self.tau = None

    
    def reset_embs(self):
        self.all_embs = {}
    
    def init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            
            if self.save:
                
                unique_id = "extr_" + str(i)

                if unique_id in self.all_embs:
                    self.all_embs[unique_id] = torch.cat((self.all_embs[unique_id], x.reshape(x.size(0), -1)), dim=0)
                else:
                    self.all_embs[unique_id] = x.reshape(x.size(0), -1)

            self.all_flatten_dims[i] = x[0].numel()

            if i == self.num_layers - 2 and self.normalize:
                x = self.tau * F.normalize(x, dim=1)

        return x


    def trim_net(self, layer, declarative_ETF=False):
       
        if layer < 0 or layer >= self.num_layers:
            raise ValueError("Layer index out of bounds.")
        
        self.net = nn.Sequential(*self.net[:layer])
        self.num_layers = layer
        self.trimmed = True
        
        if not isinstance(self.net[-1], nn.Flatten):
            self.net.append(nn.Flatten())
            self.num_layers += 1
        
        if declarative_ETF:
            print("Adding a linear projector before the declarative ETF tunnel with dim:", self.all_flatten_dims[layer-1], "->", self.num_classes)
            self.net.append(nn.Linear(self.all_flatten_dims[layer-1], self.num_classes, bias=True))
            self.num_layers += 1
            return self.num_classes
        else:
            return self.all_flatten_dims[layer-1]


    def set_etf(self, num_etf_layers=0):
        count = 0
        linear_layers = [m for m in self.net.modules() if isinstance(m, nn.Linear)]

        # Iterate over the last num_etf_layers linear layers
        for layer in linear_layers[-num_etf_layers:]:
            d_out, d_in = layer.weight.shape
            weight = torch.sqrt(torch.tensor(d_out/(d_out-1.0))) * (
                torch.eye(d_out) - (1/d_out) * torch.ones((d_out, d_out))
            )
            weight /= torch.sqrt((1/d_out) * torch.norm(weight, 'fro')**2)

            # Make sure dimensions match
            layer.weight = nn.Parameter(torch.mm(weight, torch.eye(d_out, d_in)))
            layer.weight.requires_grad_(False)
            count += 1

        print(f"Set {count} ETF layers.")


class Tunnel(nn.Module):
    def __init__(self, device, active=False, num_features=1,  normalize=False, num_classes=10, **kwargs):
        super(Tunnel, self).__init__()

        self.active = active
        self.num_classes = num_classes
        self.num_features = num_features
        self.normalize = normalize
        self.save = False
        self.tau = kwargs['temperature']
        self.decl_ETF = kwargs['declarative_ETF']
        self.ETF_fc = kwargs['ETF_fc']
        self.bias = kwargs['bias']
        self.inference = kwargs['inference']
        self.fixed_etf_layers = kwargs['fixed_etf_layers']
        self.device = device
        self.all_embs = {}
        if self.active:
            self.set_layer()
    
    def clear_layer(self):
        for attr in ['W', 'b', 'classifier', 'FeaturesMovingAverage', 'ClosestETFGeometry']:
            if hasattr(self, attr):
                delattr(self, attr)
        self.active = False
    
    def set_layer(self, verbose=True):
        if self.decl_ETF:
            if self.active and verbose:
                print("Using declarative ETF geometry.")            
            self.W = torch.zeros((self.num_classes, self.num_features))
            self.b = torch.zeros(self.num_classes)
            if not self.inference:
                self.FeaturesMovingAverage = FeaturesMovingAverageLayer(self.num_features, self.num_classes, device=self.device)
                self.ClosestETFGeometry = ClosestETFGeometryLayer(self.num_features, self.num_classes, device=self.device)
        elif self.ETF_fc:
            if self.active and verbose:
                print("Using fixed ETF geometry.")
            self.classifier = nn.Linear(self.num_features, self.num_classes, bias=True)
            self.classifier.bias.requires_grad_(False)
        else:
            if self.active and verbose:
                print("Using linear probing.")
            self.classifier = nn.Linear(self.num_features, self.num_classes, bias=self.bias)

        self.init_weights()
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.ETF_fc:
                    weight = torch.sqrt(torch.tensor(self.num_classes/(self.num_classes-1))) * (
                        torch.eye(self.num_classes)-(1/self.num_classes)*torch.ones((self.num_classes, self.num_classes))) 
                    m.weight = nn.Parameter(
                            torch.mm(weight, torch.eye(self.num_classes, self.num_features)))
                    m.weight.requires_grad_(False)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


    def get_weights(self):
        if self.decl_ETF:
            return self.W, self.b
        elif self.ETF_fc:
            return self.classifier.weight, self.classifier.bias
        else:
            bias = None if not self.bias else self.classifier.bias
            return self.classifier.weight, bias


    def activate(self, num_features, num_classes, verbose=True):
        self.active = True
        self.num_features = num_features
        self.num_classes = num_classes
        self.set_layer(verbose)
        

    def forward(self, features, y):
        
        if self.active:
            if self.normalize:
                features = self.tau * F.normalize(features, dim=1)
            if self.decl_ETF:
                if not self.inference:
                    feature_means, mu_G = self.FeaturesMovingAverage(features, y)
            
                    P = self.ClosestETFGeometry(feature_means)
                    weight = torch.sqrt(torch.tensor(self.num_classes/(self.num_classes-1))) * (torch.eye(self.num_classes)-(1/self.num_classes)*torch.ones((self.num_classes, self.num_classes)))
                    weight = weight.to(self.device)
                    
                    self.W = torch.mm(weight, P.T).to(self.device)
                    self.b = - 1.0 * torch.mv(self.W, mu_G)
                        
                    self.W.requires_grad_(True)
                    self.b.requires_grad_(True)

                x = F.linear(features, self.W, self.b)
            elif self.ETF_fc:
                self.classifier.bias.data = - 1.0 * torch.mv(self.classifier.weight, torch.mean(features, dim=0))
                x = F.linear(features, self.classifier.weight, self.classifier.bias)
            else:
                x = self.classifier(features)
            
            if self.save:
                unique_id = "tunnel"
                if unique_id in self.all_embs:
                    self.all_embs[unique_id] = torch.cat((self.all_embs[unique_id], x.reshape(x.size(0), -1)), dim=0)
                else:
                    self.all_embs[unique_id] = x.reshape(x.size(0), -1)
            return x
        else:
            return features
    
    
    def reset_embs(self):
        self.all_embs = {}



class Wrapped_NN(torch.nn.Module):
    def __init__(self, extractor, tunnel, enable_tunnel=True, normalize=False):
        super().__init__()
        self.extractor = extractor
        self.tunnel = tunnel
        self.enable_tunnel = enable_tunnel
        self.active_tunnel = tunnel.active
        self.extractor.normalize = not enable_tunnel and normalize
        self.extractor.tau = self.tunnel.tau
        self.all_embs = {}
    
    def forward(self, x, y):
        h = self.extractor(x)
        out = self.tunnel(h, y)
        return h, out

    def reset_embs(self):
        self.all_embs = {}
        self.extractor.reset_embs()
        self.tunnel.reset_embs()

    def gather_embs(self):
        self.all_embs = {**self.extractor.all_embs, **self.tunnel.all_embs}
        self.extractor.reset_embs()
        self.tunnel.reset_embs()
    
    def activate_tunnel(self, split_point, num_classes, verbose=True):
        self.active_tunnel = True
        self.extractor.normalize = False
        num_features = self.extractor.trim_net(split_point, self.tunnel.decl_ETF)
        self.tunnel.activate(num_features, num_classes, verbose)
        if self.tunnel.fixed_etf_layers>0:
            self.extractor.set_etf(self.tunnel.fixed_etf_layers)

    def get_last_layer_weights(self):
        if self.active_tunnel:
            return self.tunnel.get_weights()
        else:
            return self.extractor.net[-1].weight, self.extractor.net[-1].bias if self.extractor.net[-1].bias is not None else None