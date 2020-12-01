# Self supervised learning for digital histopathology

## Sample clustering results

<table border="0">
 <tr>
    <td><img src="https://raw.githubusercontent.com/ozanciga/self-supervised-histopathology/main/rand.png" data-canonical-src="https://raw.githubusercontent.com/ozanciga/self-supervised-histopathology/main/rand.png" width="300" height="300" /></td>
     <td><img src="https://raw.githubusercontent.com/ozanciga/self-supervised-histopathology/main/clusters.png" data-canonical-src="https://raw.githubusercontent.com/ozanciga/self-supervised-histopathology/main/clusters.png" width="300" height="300" />
 </tr>
 <tr>
</td>
   <td><b style="font-size:30px">Random sampling</b></td>
    <td>Sampling using K-means clustering of learned features</td>
 </tr>
</table>



<table border="0">
 <tr>
    <td><img src="https://raw.githubusercontent.com/ozanciga/self-supervised-histopathology/main/Selfsupunsupervisedimagesamples-1.png" width="300" height="150" /></td>
    <td><img src="https://raw.githubusercontent.com/ozanciga/self-supervised-histopathology/main/Selfsupunsupervisedimagesamples-2.png" width="300" height="150" /></td>
    <td><img src="https://raw.githubusercontent.com/ozanciga/self-supervised-histopathology/main/Selfsupunsupervisedimagesamples-3.png" width="300" height="150" /></td>
 </tr>
</table>



## Pretrained model for self supervised histopathology

You can download the model here, named **_ckpt_epoch_9.ckpt**:
https://github.com/ozanciga/self-supervised-histopathology/releases/tag/1

We save every 100th epoch, indexed starting from 0. 
This checkpoint is epoch 1000, best augmentation settings (see the paper) with Lars optimizer.

Starter code:

```
import torchvision
import torch


MODEL_PATH = '_ckpt_epoch_9.ckpt'
RETURN_PREACTIVATION = False  # return features from the model, if false return classification logits
NUM_CLASSES = 4  # only used if RETURN_PREACTIVATION = False


def load_model_weights(model, weights):

    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model


model = torchvision.models.__dict__['resnet18'](pretrained=False)

state = torch.load(MODEL_PATH, map_location='cuda:0')

state_dict = state['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

model = load_model_weights(model, state_dict)

if RETURN_PREACTIVATION:
    model.fc = torch.nn.Sequential()
else:
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.cuda()

images = torch.rand((10, 3, 224, 224), device='cuda')

out = model(images)
```

# Citation
Please use below to cite this paper if you find this repository useful or if you use pretrained models shared here in your research.

```
@article{SelfSupervisedHisto,
  title={Self supervised contrastive learning for digital histopathology},
  author={Ciga Ozan, Xu Tony, Martel Anne},
  journal={arXiv preprint arXiv:2011.13971},
  year={2020}
}
```
