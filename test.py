from backboned_unet import Unet
from backboned_unet.attention import GridAttention, AdditiveAttention, MultiplicativeImageAttention
import torch
import logging
from backboned_unet.metrics import iou
#
#from segmentation_models import Unet
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    device = 'cpu'
    if torch.cuda.is_available():
        logger.info("CUDA IS AVAILABLE")
        device = 'cuda'
    else:
        logger.info("CUDA NOT FOUND!")
    # simple test run
    input_shape = (1, 3,224, 224)
    levels_for_outputs = (2,3)
    net = Unet(backbone_name='resnet50',
               attention_module=[None, None, GridAttention, MultiplicativeImageAttention, MultiplicativeImageAttention],
               concat_with_input=False,
               input_shape=input_shape,
               levels_for_outputs=levels_for_outputs,
               classes=1)
    net.to(device)
    criterion = torch.nn.MSELoss()
    criterion1 = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    print('Network initialized. Running a test batch.')
    attentions = []
    for _ in range(1):
        with torch.set_grad_enabled(True):
            batch = torch.empty(*input_shape).normal_()
            targets = torch.empty(1, 1, *input_shape[2:]).normal_()

            out, intermediate_outputs, attention_masks = net(batch.to(device), return_attentions=True)
            attentions.append([i.detach().to('cpu').numpy() if i is not None else None for i in attention_masks])
            loss = criterion(out, targets.to(device))
            loss.backward()
            optimizer.step()
        #out = (out>0).type(torch.int64).squeeze(1)
        targets = (targets > 0).type(torch.int64).squeeze(1)
        print(targets.max(), out.max())
        print(out.shape)
        print(targets.shape)
        print(iou(out, targets))
    print('fasza.')