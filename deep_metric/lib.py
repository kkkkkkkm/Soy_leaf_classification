import torch
import torchvision


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners


class CustomDataset(Dataset):
    def __init__(self, data, train_test = True):
        super(CustomDataset, self).__init__()
        self.data = data['data']
        self.label = data['label']
        self.train_test = train_test

        if self.train_test:
            self.aug = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ConvertImageDtype(torch.float32),
                transforms.ColorJitter(brightness=0.3, hue=0.3, saturation=0.3),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.aug = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = torchvision.io.decode_image(self.data[index])
        y = self.label[index]
        
        x = self.aug(x)

        return x, y


def freeze_layers_by_block(model, freeze_blocks):
    """
    지정된 블록들의 파라미터를 고정하는 함수
    
    Args:
        model: EfficientNetB0 모델
        freeze_blocks: 고정할 블록들의 리스트
    """
    
    frozen_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        should_freeze = False
        
        # 해당 파라미터가 고정할 블록에 속하는지 확인
        for block in freeze_blocks:
            if name.startswith(block + '.'):
                should_freeze = True
                break
        
        if should_freeze:
            param.requires_grad = False
            frozen_count += 1
            print(f"고정됨: {name}")
        else:
            param.requires_grad = True
            print(f"훈련가능: {name}")

    
    print(f"\n총 {total_count}개 파라미터 중 {frozen_count}개 고정됨")
    print("-" * 50)
    return model


def freeze_layers_by_percentage(model, freeze_ratio = 0.7):
    ## 모델의 레이어 수를 기준으로 가중치 고정
    
    feature_blocks = []

    for name, module in model.features.named_children():
        feature_blocks.append(f"features.{name}")
    
    freeze_count = int(len(feature_blocks) * freeze_ratio)
    freeze_blocks = feature_blocks[:freeze_count]

    print(f"Features 블록 총 {len(feature_blocks)}개 중 {freeze_count}개 고정")
    print(f"고정할 블록들: {freeze_blocks}")

    return freeze_layers_by_block(model, freeze_blocks)




def freeze_parameters_by_percentage(model, freeze_ratio = 0.7):
    ##전체 파라미터의 수를 기준으로 가중치 고정
    
    all_params = list(model.parameters())
    total_count = 0
    frozen_count = 0
    num_params_to_freeze = int(len(all_params) * freeze_ratio)

    for idx, (name, param) in enumerate(model.named_parameters()):
        total_count += 1
        if idx < num_params_to_freeze:
            frozen_count +=1
            param.requires_grad = False
            print(f"고정됨: {name}")
        else:
            param.requires_grad = True
            print(f"훈련가능: {name}")

    print(f"\n총 {total_count}개 파라미터 중 {frozen_count}개 고정됨")
    print("-" * 50)
    
    return model



def metric_train(train_loader, model, loss_fn, optimizer, miner_func, device = torch.device('cuda')):
    total_loss = 0
    model.train()
    for data, labels in tqdm(train_loader, desc = 'Train'):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(data)
        miner_output = miner_func(embeddings, labels)
        
        loss = loss_fn(embeddings, labels, miner_output)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(train_loader)


def metric_validation(valid_loader, model, loss_fn, miner_func, device = torch.device('cuda')):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, labels in tqdm(valid_loader, desc = 'Valid'):
            data = data.to(device)
            labels = labels.to(device)
            
            embeddings = model(data)
            miner_output = miner_func(embeddings, labels)
        
            loss = loss_fn(embeddings, labels, miner_output)
            
            total_loss += loss.item()

    return total_loss / len(valid_loader)



def classifier_train(train_loader, model, loss_fn, optimizer, device = torch.device('cuda')):
    total_loss, correct = 0, 0
    model.train()
    
    for data, labels in tqdm(train_loader, desc = 'Train'):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        _, pred = model(data)
                
        loss = loss_fn(pred, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    total_loss /= len(train_loader)
    correct /= len(train_loader.dataset)
    
    return total_loss, correct


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def classifier_validation(valid_loader, model, loss_fn, device= torch.device('cuda')):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for data, labels in tqdm(valid_loader, desc = 'Valid'):
            data = data.to(device)
            labels = labels.to(device)
            
            _, pred = model(data)
        
            loss = loss_fn(pred, labels)
            total_loss += loss.item()

            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    
    total_loss /= len(valid_loader)
    correct /= len(valid_loader.dataset)
    
    return total_loss, correct

