import torch

def l2_diff(net1, net2):
    total = 0
    net1_params = {}
    l2_params = {}
    for name, param in net1.named_parameters():
        net1_params[name] = param
        
    for name, param in net2.named_parameters():
        orig =  net1_params[name]
        L2 = torch.norm(orig - param, 2)
        l2_params[name] = L2.item()
        total += L2 *L2

    total = torch.sqrt(total)
    return total.item(), sorted(l2_params.items(), key=lambda x:x[1], reverse=True)