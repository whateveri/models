
import os
import torch
import data_setup,eigine,utils,model_builder

from torchvision import transforms

NUM_EPOCHS=5
BATCH_SIZE=32
HIDDEN_UNITS=10
lr=0.001

train_dir="data/pizza_steak_sushi/train"
test_dir="data/pizza_steak_sushi/test"

device="cuda" if torch.cuda.is_available() else "cpu"

data_trainsform=transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])


train_dataloader,test_dataloader,class_names=data_setup.create_dataloaders(                      train_dir=train_dir,
                       test_dir=test_dir,
                       train_transform=data_trainsform,
                       test_transform=data_trainsform,
                       batch_size=BATCH_SIZE
                       )

model=model_builder.TinyVGG(input_features=3,
                 hidden_features=HIDDEN_UNITS,
                 output_features=len(class_names)).to(device)

loss_fn=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(params=model.parameters(),
                           lr=lr)

eigine.train(model=model,
             train_loader=train_dataloader,
             test_loader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

utils.save_model(model=model,
               path="models",
               model_name="go_script_TinyVGG")



