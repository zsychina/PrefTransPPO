import torch
import torch.optim as optim
import torch.nn as nn

from dataset import preference_dataset, traj_target_length
from transformer import Transformer

import matplotlib.pyplot as plt



# Training hyperparameters
batch_size = 128
learning_rate = 1e-4
epochs = 50
state_dim = preference_dataset[0]['observations_1'].shape[-1]
action_dim = preference_dataset[0]['actions_1'].shape[-1]

# Split dataset into train and validation sets
train_size = int(0.8 * len(preference_dataset))
val_size = len(preference_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(preference_dataset, [train_size, val_size])

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = Transformer(n_layer=3,
                    in_dim=state_dim+action_dim, 
                    out_dim=1, 
                    n_head=8,
                    dim=256, 
                    dropout=0.3, 
                    max_seqlen=traj_target_length+1).to(device)



# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True
)


val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Before the training loop, add these lists to store metrics:
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        traj1_observations = batch['observations_1']
        traj2_observations = batch['observations_2']
        
        traj1_actions = batch['actions_1'] 
        traj2_actions = batch['actions_2']
        
        traj1 = torch.cat([traj1_observations, traj1_actions], dim=-1).to(device)
        traj2 = torch.cat([traj2_observations, traj2_actions], dim=-1).to(device)
        
        traj1_rewards = model(traj1)
        traj2_rewards = model(traj2)

        traj1_rewards_sum = traj1_rewards.sum(dim=1)  # Shape: [batch_size, 1]
        traj2_rewards_sum = traj2_rewards.sum(dim=1)  # Shape: [batch_size, 1]
        
        logits = torch.cat([traj1_rewards_sum, traj2_rewards_sum], dim=-1) 
        
        labels = batch['preference'].to(device)
        
        # Compute cross entropy loss between logits and labels
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            traj1_observations = batch['observations_1']
            traj2_observations = batch['observations_2']
            
            traj1_actions = batch['actions_1'] 
            traj2_actions = batch['actions_2']
            
            traj1 = torch.cat([traj1_observations, traj1_actions], dim=-1).to(device)
            traj2 = torch.cat([traj2_observations, traj2_actions], dim=-1).to(device)
            
            traj1_rewards = model(traj1)
            traj2_rewards = model(traj2)
            
            traj1_rewards_sum = traj1_rewards.sum(dim=1)
            traj2_rewards_sum = traj2_rewards.sum(dim=1)
            
            logits = torch.cat([traj1_rewards_sum, traj2_rewards_sum], dim=-1)
            
            labels = batch['preference'].to(device)
            
            val_loss += criterion(logits, labels).item()
            pred = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    # Store the metrics
    train_losses.append(train_loss/len(train_loader))
    val_losses.append(val_loss/len(val_loader))
    val_accuracies.append(100*correct/total)
    
    # Step the scheduler based on validation loss
    scheduler.step(val_loss/len(val_loader))
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f'Epoch {epoch+1}/{epochs}:')
    print(f'Training Loss: {train_losses[-1]:.4f}')
    print(f'Validation Loss: {val_losses[-1]:.4f}')
    print(f'Validation Accuracy: {val_accuracies[-1]:.2f}%')
    print(f'Learning Rate: {current_lr:.6f}\n')

    # Save model parameters
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'checkpoints/reward_model_epoch_{epoch+1}.pt')
        print(f'Saved model checkpoint at epoch {epoch+1}')

# After training loop, add plotting code:
plt.figure(figsize=(12, 4))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plots.png')
plt.close()




