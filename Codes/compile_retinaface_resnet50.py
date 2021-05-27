from models.retinaface import RetinaFace
import torch
from os import cpu_count, path
from data import cfg_re50
import torch_neuron


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(
        pretrained_path, map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    # Use ResNet50 backbone and pre-trained weights
    cfg = cfg_re50
    trained_model_pth = './weights/Resnet50_Final.pth'
    trained_model_neuron = './weights/Resnet50_Final_neuron.pt'
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, trained_model_pth)
    net.eval()
    print('Finished loading model!')

    # input shape from https://t.corp.amazon.com/P47075957/
    image = torch.rand([1, 3, 432, 768])

    # Run example CPU inference
    cpu_output = net(image)

    # Compile the model using AWS Neuron
    neuron_net = torch_neuron.trace(net, image)
    print('\nSuccessfully compiled the RetinaFace model\n')

    # Run test inference with compiled model
    neuron_output = neuron_net(image)

    # Save the compiled model
    torch.jit.save(neuron_net, trained_model_neuron)

    print('\nSuccessfully ran inference with the RetinaFace model\n')
