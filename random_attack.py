from utils import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = Model().to(device)
#model = torch.load('./models/mnistcuangai0407-mod3').to(device)
#model = torch.load('models/resnet18/cifar10/cifar10_0403_0.7158-exchange-adapt-mod511.pkl').to(device)
model = torch.load('./models/resnet50/cifar10/cifar10_0403_0.7432-exchange-adapt-mod511.pkl').to(device)


alpha = 0.27
for _,parmeters in model.named_parameters():    #parmeters也就是一层的weight或者bias的参数
    a, b = flattenM(parmeters)
    lenth = len(a)
    random_ints = random.sample(range(lenth), math.floor(lenth*alpha))
    with torch.no_grad():

        for i,j in enumerate(a):
            if i in random_ints:
                print(a[i])

                a[i:i+1] = torch.tensor(random.random()*1)
                print(a[i])

    break


torch.save(model, './models/resnet50/cifar10/cifar10_0403_0.7432-exchange-adapt-teshu-attack-0.01.pkl')