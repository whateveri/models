import torch
from tqdm.auto import tqdm
from typing import List,Dict,Tuple


def train_step(model:nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:nn.Module,
               optimizer:torch.optim.Optimizer,
               device:torch.device)-> Tuple[float,float]:
    model.train()

    train_loss, train_acc=0.0,0.0

    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)

        y_pred_logit=model(X)

        loss=loss_fn(y_pred_logit,y)

        train_loss+=loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class=torch.argmax(torch.softmax(y_pred_logit,dim=1),dim=1)

        train_acc+=((y_pred_class==y).sum().item() / len(y_pred_logit))
    
    train_loss=train_loss/len(dataloader)
    train_acc=train_acc/len(dataloader)

    return train_loss,train_acc


def test_step(model:nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn:nn.Module,
              optimizer:torch.optim.Optimizer,
              device:torch.device)->Tuple[float,float]:
    model.eval()
    test_loss,test_acc=0.0,0.0

    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X,y=X.to(device),y.to(device)

            y_pred_logit=model(X)

            loss=loss_fn(y_pred_logit,y)

            test_loss+=loss.item()

            y_pred_label=torch.argmax(y_pred_logit,dim=1)

            test_acc+=((y_pred_label==y).sum().item() / len(y_pred_logit))

    test_loss=test_loss / len(dataloader)
    test_acc=test_acc / len(dataloader)
    
    return test_loss,test_acc


def train(model:nn.Module,
          train_loader:torch.utils.data.DataLoader,
          test_loader:torch.utils.data.DataLoader,
          loss_fn:nn.Module,
          optimizer:torch.optim.Optimizer,
          epochs:int,
          device:torch.device
          )-> Dict[str,int]:

    results={
        "train_losses":[],
        "train_acces":[],
        "test_losses":[],
        "test_acces":[]
    }

    for epoch in tqdm(range(epochs)):

        train_loss,train_acc=train_step(model=model,
                                            dataloader=train_loader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        
        test_loss,test_acc=test_step(model=model,
                                         dataloader=test_loader,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         device=device)
        print(f'Epoch {epoch+1} |'
              f'Train loss: {train_loss}'
              f'Train acc: {train_acc}'
              f'Test loss: {test_loss}'
              f'Test acc: {test_acc}')
        results["train_losses"].append(train_loss)
        results["train_acces"].append(train_acc)
        results["train_losses"].append(test_loss)
        results["train_losses"].append(test_acc)

    return results




            



    

        

            
