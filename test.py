from backboned_unet import Unet
from backboned_unet.attention import GridAttention, AdditiveAttention, MultiplicativeImageAttention
import torch
#from segmentation_models import Unet
if __name__ == "__main__":

    # simple test run
    input_shape = (3, 3, 128, 128)
    net = Unet(backbone_name='resnet50', attention_module=MultiplicativeImageAttention, concat_with_input=True, input_shape=input_shape)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    print('Network initialized. Running a test batch.')
    for _ in range(1):
        with torch.set_grad_enabled(True):
            batch = torch.empty(*input_shape).normal_()
            targets = torch.empty(1, 21, *input_shape[2:]).normal_()

            out = net(batch)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        print(out.shape)

    print('fasza.')