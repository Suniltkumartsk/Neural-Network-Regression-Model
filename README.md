# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SUNIL KUMAR T
### Register Number: 212223240164
```python
import torch.nn as nn
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(1,8)
    self.fc2=nn.Linear(8,10)
    self.fc3=nn.Linear(10,1)
    self.relu=nn.ReLU()
    self.history={'loss':[]}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x


# Initialize the Model, Loss Function, and Optimizer
leo = NeuralNet()
criterion=nn.MSELoss()
optimizer=torch.optim.RMSprop(leo.parameters(),lr=0.001)


def train_model(leo,X_train,y_train,criterion,optimizer,epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(leo(X_train),y_train)
    loss.backward()
    optimizer.step()

    leo.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

train_model(leo,X_train_tensor,Y_train_tensor,criterion,optimizer)
with torch.no_grad():
  test_loss=criterion(leo(X_test_tensor),Y_test_tensor)
  print(f"Test loss: {test_loss.item():.6f}")

import matplotlib.pyplot as plt
plt.plot(leo.history['loss'])
plt.title("Loss curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")



```
## Dataset Information
<img width="694" height="434" alt="Screenshot 2025-08-19 111205" src="https://github.com/user-attachments/assets/0c2fb7d1-c0ad-4684-8cc8-6be5c6660b8d" />

## OUTPUT

### Training Loss Vs Iteration Plot
<img width="939" height="501" alt="Screenshot 2025-08-19 111219" src="https://github.com/user-attachments/assets/40ee4f47-522c-413f-a7c1-99c95ce77b0f" />



## RESULT
To develop a neural network regression model for the given dataset is excuted sucessfully.
