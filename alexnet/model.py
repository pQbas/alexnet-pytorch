import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, categories=10):
        super(AlexNet, self).__init__()

        self.stack1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.stack2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.stack3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.stack4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.stack5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3,3)
        )

        self.stack6 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU()
        )

        self.stack7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU()
        )

        self.stack8 = nn.Sequential(
            nn.Linear(4096, categories),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        x = self.stack5(x)
        x = torch.flatten(x)
        x = self.stack6(x)
        x = self.stack7(x)
        x = self.stack8(x)
        return x


if __name__ == '__main__':
    print('return x')

    model = AlexNet(categories=10)

    input  = torch.rand([1, 3, 224, 224])
    output = model(input)
    print(output.shape)


