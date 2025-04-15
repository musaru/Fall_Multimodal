from .st_gcn.stgcan import STGCAN
from .bilstm import BiLSTM
from .combination import TwoStreamSTGCAN, TwoStreamSTGCAN_BiLSTM

def build_model(config):
    model_name = config.MODEL.NAME
    
    if model_name == "stgcn":
        model = STGCAN(config.DATA.IN_CHANNELS, {'layout':config.GRAPH.LAYOUT,'strategy': config.GRAPH.STRATEGY}, num_class=config.DATA.NUM_CLASSES)
    elif model_name == "bilstm":
        model = BiLSTM(input_size=config.DATA.SENSOR_DIM,hidden_size=64,num_layers=1,dropout_prob=0.3,num_classes=config.DATA.NUM_CLASSES,feature="mean")
    elif model_name == "two_stgcan":
        model = TwoStreamSTGCAN(config.DATA.IN_CHANNELS, {'layout':config.GRAPH.LAYOUT,'strategy': config.GRAPH.STRATEGY}, num_class=config.DATA.NUM_CLASSES)
    elif model_name == "two_stgcan_bilstm":
        model = TwoStreamSTGCAN_BiLSTM(config.DATA.IN_CHANNELS, {'layout':config.GRAPH.LAYOUT,'strategy': config.GRAPH.STRATEGY}, num_class=config.DATA.NUM_CLASSES, bilstm_input_size=config.DATA.SENSOR_DIM)
    else:
        raise RuntimeError(f"Model name [{model_name}] is not implemented.")
        
    return model