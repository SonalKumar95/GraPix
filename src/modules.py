import torch
from utils import *
import torch.nn.functional as F
import dino.vision_transformer as vits

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        patch_size = self.cfg.dino_patch_size
        self.patch_size = patch_size
        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model_type
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=.1)

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if cfg.pretrained_weights is not None:
            state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
            state_dict = state_dict["teacher"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.pretrained_weights, msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)
        
        return image_feat

class ResizeAndClassify(nn.Module):

    def __init__(self, dim: int, size: int, n_classes: int):
        super(ResizeAndClassify, self).__init__()
        self.size = size
        self.predictor = torch.nn.Sequential(
            torch.nn.Conv2d(dim, n_classes, (1, 1)),
            torch.nn.LogSoftmax(1))

    def forward(self, x):
        return F.interpolate(self.predictor.forward(x), self.size, mode="bilinear", align_corners=False)
                                   
class SelfLabelLookup(nn.Module):
    def __init__(self, threshold = 0.99, apply_class_balancing = True):
        super(SelfLabelLookup, self).__init__()
         
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing
        self.true_samples = []
        self.false_samples = []
          
    def MaskedCrossEntropyLoss(self, input_, target_, mask_, weight_, reduction='mean'):
        if not (mask_ != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        
        target_ = torch.masked_select(target_, mask_)
        b_, c_ = input_.size()
        n_ = target_.size(0)
        input_ = torch.masked_select(input_, mask_.view(b_, 1)).view(n_, c_)
        return F.cross_entropy(input_, target_, weight = weight_, reduction = reduction)

    def forward(self, anchors_weak, anchors_strong=None, alpha=None, log_probs=False, norm=False):
        
        if norm:
            anchors_weak = F.normalize(anchors_weak, dim=1)
            if anchors_strong is not None:
                anchors_strong = F.normalize(anchors_strong, dim=1)
        
        if alpha is not None:
            weak_anchors_prob = nn.functional.softmax(anchors_weak * alpha, dim=1)
        else:
            weak_anchors_prob = nn.functional.softmax(anchors_weak, dim=1)
        
        B_,C_,H_,W_ = weak_anchors_prob.shape
        weak_anchors_prob = weak_anchors_prob.permute(0, 3, 2, 1).reshape(B_*H_ * W_, C_)
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        
        anchors_strong = anchors_strong.permute(0, 3, 2, 1).reshape(B_ * H_ * W_, C_)
            
        #print(max_prob)
        mask = max_prob > self.threshold 
        xx, count = torch.unique(mask, return_counts = True)

        try :
            #print(xx,count)
            self.true_samples.append(int(count[1])/(int(count[1])+int(count[0])))
            self.false_samples.append(int(count[0])/(int(count[1])+int(count[0])))
        except: #IN CASE OF ALL SAMPLE BELONGS TO ONE TYPE(T/F)
            self.true_samples.append(0)
            self.false_samples.append(0)                    
        #print(xx,count)

        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)
            
        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            #print(idx,counts)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c,device=anchors_strong.device)
            weight[idx] = freq

        else:
            weight = None
            
        # Loss
        loss = self.MaskedCrossEntropyLoss(anchors_strong, target, mask, weight, reduction='mean') 
        return loss   
                      
class KNNLookup(nn.Module):
    def __init__(self, topk = 10, entropy_weight = 2.0):
        super(KNNLookup, self).__init__()
        #self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0
        self.topk = topk
        self.EPS = 1e-8
        print('-----------------------------KNNLookup----------------------')

    def entropy(self,x, input_as_probabilities):
        if input_as_probabilities:
            x_ =  torch.clamp(x, min = self.EPS)
            b =  x_ * torch.log(x_)
        else:
            b = nn.functional.softmax(x, dim = 1) * nn.functional.log_softmax(x, dim = 1)

        if len(b.size()) == 2: # Sample-wise entropy
            return -b.sum(dim = 1).mean()
        elif len(b.size()) == 1: # Distribution-wise entropy
            return - b.sum()
        else:
            raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

    def forward(self, anchors, prob):
        # Softmax
        b,c,h,w = anchors.shape
        anchors = anchors.permute(0, 3, 2, 1).reshape(b * h * w, c)
        prob = prob.permute(0, 3, 2, 1).reshape(b * h * w, -1)
        
        n = prob.shape[1]

        feats = (anchors @ anchors.t()).squeeze()
        top_val,top_ind = torch.topk(feats,self.topk+1,largest=True, sorted=True)
        
        anchors_prob = nn.functional.softmax(prob, dim=1)
        positives_prob = anchors_prob[top_ind[:,1:]]
         
        similarity = torch.bmm(anchors_prob.view(b * h * w, 1, n), positives_prob.view(b * h * w, n, self.topk)).squeeze()
        ones = torch.ones_like(similarity, device=similarity.device)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = self.entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss
        
class GraphLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(GraphLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))
        self.resolution = 0.05
        print('-----------------------------GraphLookup:{}----------------------'.format(self.resolution))
    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha=torch.tensor(1), log_probs=False):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        B, C, H, W  = normed_features.shape

        merged_features = normed_features.permute(0, 3, 2, 1).reshape(B * H * W, C)
        similarities =  (torch.matmul(merged_features, merged_features.t())).squeeze()
        similarities_pos = torch.clamp(similarities, min=0) # SIZE: [(batch * #patch), (batch * #patch)]
        similarities_pos = similarities_pos - similarities_pos * torch.eye(similarities_pos.shape[1], device=similarities.device)

        Deg = torch.sum(similarities_pos, dim=1).unsqueeze(1)
        norm = torch.sum(Deg)/2
        Q = similarities_pos - (self.resolution * (torch.matmul(Deg,Deg.t())/(2*norm)))
 
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
        
        if alpha is not None:
            soft_probs = nn.functional.softmax(inner_products * alpha, dim=1)
        else:
            soft_probs = nn.functional.softmax(inner_products, dim=1)
        B_, C_, H_, W_  = soft_probs.shape
        soft_products = soft_probs.permute(0, 3, 2, 1).reshape(B_ * H_ * W_, C_) # SIZE: [(batch * #patch), #Clusters]
         
        hard_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]).to(torch.float32)
         
        if alpha is None:
            cluster_probs = hard_probs.permute(0, 3, 1, 2)
        else:
            cluster_probs = nn.functional.log_softmax(inner_products * alpha, dim=1)
             
        soft_modularity = torch.matmul((torch.matmul(soft_products.t(),Q)),soft_products)
        soft_modularity = soft_modularity.diagonal(offset=0, dim1=-2, dim2=-1)
        soft_modularity_loss = -(soft_modularity.sum(dim=-1))/(2*norm)
         
        cluster_loss = -(hard_probs.permute(0, 3, 1, 2) * inner_products).sum(1).mean()
                
        # orthogonality loss
        St_S = torch.matmul(soft_products.t(), soft_products)
        I_S = torch.eye(self.n_classes, device=St_S.device)
        ortho_loss = torch.norm(St_S / torch.norm(St_S) - I_S/ torch.norm(I_S))

        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return soft_modularity_loss + cluster_loss + ortho_loss , soft_modularity_loss , cluster_probs
             
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

def average_norm(t):
    return t / t.square().sum(1, keepdim=True).sqrt().mean()

def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)

def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)

@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size

def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])