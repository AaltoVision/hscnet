from .scrnet import SCRNet
from .hscnet import HSCNet

def get_model(name, dataset):
    return {
            'scrnet' : SCRNet(),
            'hscnet' : HSCNet(dataset=dataset)    
           }[name]

