from backboned_unet import Unet
from backboned_unet.attention import GridAttention, AdditiveAttention, MultiplicativeImageAttention
import torch
#from segmentation_models import Unet
if __name__ == "__main__":
    device = 'cpu'
    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")
        device = 'cuda'
    # simple test run
    input_shape = (1, 3,96, 96)
    net = Unet(backbone_name='resnet50',
               attention_module=GridAttention,
               concat_with_input=True,
               input_shape=input_shape,
               classes=1)
    net.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    print('Network initialized. Running a test batch.')
    for _ in range(1):
        with torch.set_grad_enabled(True):
            batch = torch.empty(*input_shape).normal_()
            targets = torch.empty(1, 21, *input_shape[2:]).normal_()

            out = net(batch.to(device))
            loss = criterion(out, targets.to(device))
            loss.backward()
            optimizer.step()
        print(out.shape)

    print('fasza.')