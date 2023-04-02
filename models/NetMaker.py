from .models import DADFNet
def model_bulider(config):
    model_type = config['model_type']
    embeading_dim = config['embeading_dim']
    stage = config['stage']
    num_blocks = config['num_blocks']
    img_size = config['img_size']
    Multiscale = config['Multiscale']
    type = config['Type']
    input_type = config['inType']
    n_s = config['n_s']
    if model_type == 'DADFNet':
        model = DADFNet(embeading_dim,stage,n_s,num_blocks,image_size=img_size, Multiscale=Multiscale,type=type,input_fusion=input_type)
    return model