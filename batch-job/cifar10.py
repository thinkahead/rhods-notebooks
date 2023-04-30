# Copyright 2022 IBM, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# In[]
import os

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
# %%

print("prior to running the trainer")
print("MASTER_ADDR: is ", os.getenv("MASTER_ADDR"))
print("MASTER_PORT: is ", os.getenv("MASTER_PORT"))


class LitCIFAR10(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (3, 32, 32)
        self.example_input_array = torch.randn(1, 3, 32, 32)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [   transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124),
                                     std=(0.24703233, 0.24348505, 0.26158768))
            ]
        )
        
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()        
        
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(64,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(128,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = nn.Linear(64*4*4,4096)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(4096, 1024)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, 10)    
    
    def forward(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = x.view(-1,64*4*4)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        
        return F.log_softmax(x,dim=1)    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        print("Downloading CIFAR10 dataset...")
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45000, 5000])
            #self.cifar10_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            #self.cifar10_val = CIFAR10(self.data_dir, train=False, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=BATCH_SIZE)


# Init DataLoader from CIFAR10 Dataset

model = LitCIFAR10()

print("GROUP: ", int(os.environ.get("GROUP_WORLD_SIZE", 1)))
print("LOCAL: ", int(os.environ.get("LOCAL_WORLD_SIZE", 1)))

from pytorch_lightning.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

from pytorch_lightning.callbacks import ModelCheckpoint
model_checkpoint_path = '/tmp'
checkpoint_callback = ModelCheckpoint(monitor='val_loss', #val_loss is default metric
                                      dirpath = model_checkpoint_path,
                                      filename = 'cifar10_scheduler-{epoch:02d}-{val_loss:2f}',
                                      save_top_k=1, # How many top models to save
                                      mode='min')
                                                                                           
# Initialize a trainer
trainer = Trainer(
    accelerator="auto",
    # devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=100,
    callbacks=[early_stopping,checkpoint_callback,TQDMProgressBar(refresh_rate=20)],
    num_nodes=int(os.environ.get("GROUP_WORLD_SIZE", 1)),
    devices=int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
    strategy="ddp",
)

# Train the model ⚡
trainer.fit(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
#torch.save(model, '/tmp/model.pt')

dummy_input = torch.randn(1, 3, 32, 32).to(device)
input_names = [ "input_0" ]
output_names = [ "output_0" ]
dynamic_axes={'input_0' : {0 : 'batch_size'},'output_0' : {0 : 'batch_size'}}

#model.to_onnx('/tmp/cifar10-4.onnx', dummy_input, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
torch.onnx.export(model, dummy_input, '/tmp/cifar10.onnx', verbose=True, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
#model.to_onnx('/tmp/cifar10-2.onnx', dummy_input, input_names=input_names, output_names=output_names)
#torch.onnx.export(model, dummy_input, '/tmp/cifar10-1.onnx', verbose=True, input_names=input_names, output_names=output_names)

print("GLOBAL_RANK: is ", trainer.global_rank)
if trainer.global_rank==0:
    modelfile='/tmp/cifar10.onnx'
    print("Copying",modelfile)
    import os
    import boto3
    from boto3 import session

    key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    uploaded_file_name = os.environ.get('OUTPUT_PATH',os.uname()[1])
    session = boto3.session.Session(aws_access_key_id=key_id, aws_secret_access_key=secret_key)
    s3_client = boto3.client('s3', aws_access_key_id=key_id, aws_secret_access_key=secret_key,endpoint_url=endpoint_url,verify=False)
    buckets=s3_client.list_buckets()
    for bucket in buckets['Buckets']: print(bucket['Name'])
    s3_client.upload_file(modelfile, bucket['Name'],uploaded_file_name)
    print('uploaded_file_name',uploaded_file_name)
    print([item.get("Key") for item in s3_client.list_objects_v2(Bucket=bucket['Name']).get("Contents")])
    #s3_client.delete_object(Bucket=bucket['Name'],Key='mmm.onnx')
    # You can pass `test(ckpt_path='best')` to use best model checkpoint or
    #    `ckpt_path=trainer.checkpoint_callback.last_model_path` to use the last model
    #result = trainer.test(trainer.checkpoint_callback.last_model_path)
    #print(result)

