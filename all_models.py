import torch
import torch.nn as nn
import torchvision
from nets.unet_model import UNet
from nets.uInterp_model import uInterp, uInterp_topdown, uInterp_multistream
# FOR CERTIFICATE ISSUES
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# from torchvision.models.segmentation.segmentation import model_urls as segmentation_urls
# segmentation_urls['fcn_resnet101_coco'] = segmentation_urls['fcn_resnet101_coco'].replace('https://', 'http://')
# segmentation_urls['deeplabv3_resnet101_coco'] = segmentation_urls['deeplabv3_resnet101_coco'].replace('https://', 'http://')

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.base_net = torchvision.models.resnet18(pretrained=True)
        self.num_classes = num_classes
        self.base_net.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_net(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50, self).__init__()
        self.base_net = torchvision.models.resnet50(pretrained=True)
        self.num_classes = num_classes
        self.base_net.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base_net(x)
        return x

class UNetMirc(nn.Module):
    def __init__(self, num_parts=5):
        super(UNetMirc, self).__init__()
        self.num_parts = num_parts
        self.base_net = UNet(n_channels=3, n_classes=self.num_parts)

    def forward(self, x):
        x = self.base_net(x)
        return x

class FCNMirc(nn.Module):
    def __init__(self, num_parts=5):
        super(FCNMirc, self).__init__()
        self.num_parts = num_parts
        self.base_net = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
        #self.base_net = torchvision.models.segmentation.fcn_resnet50()
        self.base_net.classifier[4] = nn.Conv2d(
            in_channels=512,
            out_channels=self.num_parts,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):
        x = self.base_net(x)
        return x['out']

class DeepLabMirc(nn.Module):
    def __init__(self, num_parts=5):
        super(DeepLabMirc, self).__init__()
        self.num_parts = num_parts
        self.base_net = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        #self.base_net = torchvision.models.segmentation.deeplabv3_resnet50()
        self.base_net.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=self.num_parts,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):
        x = self.base_net(x)
        return x['out']

class RecUNetMirc(nn.Module):
    def __init__(self, loss_ratio=1, num_parts=5, topdown_class=False, multistream_class=False):
        super(RecUNetMirc, self).__init__()
        self.loss_ratio = loss_ratio
        self.num_parts = num_parts
        self.num_classes = 2
        self.topdown_class = topdown_class
        self.multistream_class = multistream_class
        if self.topdown_class:
            self.interp_net = uInterp_topdown(n_channels=3, n_classes=self.num_parts)
        elif self.multistream_class:
            self.interp_net = uInterp_multistream(n_channels=3, n_classes=self.num_parts)
        else:
            self.interp_net = uInterp(n_channels=3, n_classes=self.num_parts)
        self.class_net = nn.Sequential(
            nn.Dropout(),
            nn.Linear(576, 128),  # 3136
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        interp_out, features = self.interp_net(x)
        class_out = self.class_net(features)
        return class_out, interp_out


def load_model(model_name, loss_ratio=[1, 1], num_internal_parts=5):
    print('loading network model: %s..' % model_name)

    if model_name == 'ResNet18':
        model = ResNet18()
        model.num_parts = 0
        model.output_type = 'img2class'

    elif model_name == 'ResNet50':
        model = ResNet50()
        model.num_parts = 0
        model.output_type = 'img2class'

    elif model_name == 'UNetMirc':
        model = UNetMirc(num_parts=num_internal_parts + 1)  # TODO Dice: change to num_parts=num_internal_parts
        model.output_type = 'img2intep'

    elif model_name == 'DeepLabMirc':
        model = DeepLabMirc(num_parts=num_internal_parts + 1)
        model.output_type = 'img2intep'

    elif model_name == 'FCNMirc':
        model = FCNMirc(num_parts=num_internal_parts + 1)
        model.output_type = 'img2intep'

    elif model_name == 'RecUNetMirc':
        model = RecUNetMirc(loss_ratio=loss_ratio, num_parts=num_internal_parts+1)
        model.output_type = 'img2dual'

    elif model_name == 'RecUNetMircTD':
        model = RecUNetMirc(loss_ratio=loss_ratio, num_parts=num_internal_parts+1, topdown_class=True)
        model.output_type = 'img2dual'

    elif model_name == 'RecUNetMircMulti':
        model = RecUNetMirc(loss_ratio=loss_ratio, num_parts=num_internal_parts+1, topdown_class=False, multistream_class=True)
        model.output_type = 'img2dual'

    else:
        print('ERROR: model name does not exist..')
        return

    return model


if __name__ == '__main__':
    img = torch.rand(10, 3, 108, 108)
    m = load_model('ResNet50')
    #m = load_model('RecUNetMircTD')
    #m = load_model('RecUNetMircMulti')
    #m = load_model('UNetMirc')
    #m = load_model('DeepLabMirc')
    #m = load_model('FCNMirc')
    img.requires_grad_()
    m.train()
    o = m(img)

    # interp_gt = torch.randint(low=0, high=6, size=(1, 108, 108)).long()  # torch.rand(1, 1)
    # cri = load_loss('Dice', num_parts=5+1)
    # loss = cri(m(img), interp_gt)
    # optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    # loss.backward()
    # optimizer.step()
    # print('hi')
    pass
    # t = torch.rand(1, 1)
    # a = MetaResCoordiNet50()
    # print(a(img, t))
    # print(a.train_meta(img, t, coords, l2loss)[0])
    # optimizer = torch.optim.Adam(a.resnet.parameters(), lr=1e-4)
    # optimizer.step()
    # print(a(img, t))
