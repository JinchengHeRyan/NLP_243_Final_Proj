import torch
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch.nn.functional as F
# 1. Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load the pre-trained GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("semeval25-unlearning-model",attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16 ).to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
oracle_model = AutoModelForCausalLM.from_pretrained("semeval25-unlearning-model",attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16 ).to(device)
# 3. Set the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 4. Function to compute loss (cross-entropy loss for language modeling)
def compute_forget_loss(model, inputs, targets):
    inputs = inputs.to(device)
    targets = targets.to(device)
    attention_mask = torch.ones_like((inputs != 0))
    outputs = model(inputs, attention_mask=attention_mask, labels=targets)
    output = outputs.logits
    def batch_loss(targets, output):
        shifted_labels = targets[..., 1:].contiguous()
        output = output[..., :-1, :].contiguous()
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
        loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
        return loss
    # get the sum loss for each sequence in a batch
    loss = batch_loss(targets, output)
    loss1 = batch_loss(targets, oracle_model(inputs, attention_mask=attention_mask, labels=targets).logits)
    loss = loss - loss1
    beta =0.1
    gamma = 0.1
    return -F.logsigmoid(beta * loss).mean() * 2 / beta  - gamma



def compute_loss(model, inputs, targets,type="retain"):
    if(type=="forget"):
        return compute_forget_loss(model, inputs, targets)
    # Move inputs and targets to the same device as the model
    inputs = inputs.to(device)
    targets = targets.to(device)
    attention_mask = torch.ones_like((inputs != 0))
    # Forward pass through the model
    outputs = model(inputs, labels=targets,attention_mask=attention_mask)
    # Extract the loss (language modeling loss is included in outputs)
    loss = outputs.loss
    return loss
# 5. Function to update the model with gradient ascent (forgetting)
def forget_set_update(model, optimizer, forget_inputs, forget_outputs):
    loss_values = []
    for input_ids, target_ids in zip(forget_inputs, forget_outputs):
        optimizer.zero_grad()
        # print("input_ids",input_ids.shape)
        
        # Compute loss using the forget set
        loss = compute_loss(model, input_ids, target_ids,type="forget")
        
        # Perform backward pass to calculate gradients
        loss.backward()
        
        # # Flip the gradient to perform gradient ascent
        # for param in model.parameters():
        #     param.grad = -param.grad
        
        # Apply the update
        optimizer.step()
        
        # Track loss for plotting
        loss_values.append(loss.item())
    
    return loss_values

# 6. Function to update the model with gradient descent (retaining)
def retain_set_update(model, optimizer, retain_inputs, retain_outputs):
    loss_values = []
    for input_ids, target_ids in zip(retain_inputs, retain_outputs):
        optimizer.zero_grad()
        
        # Compute loss using the retain set
        loss = compute_loss(model, input_ids, target_ids)
        
        # Perform backward pass to calculate gradients
        loss.backward()
        
        # Apply the update (gradient descent)
        optimizer.step()
        
        # Track loss for plotting
        loss_values.append(loss.item())
    
    return loss_values

# 7. Tokenizer utility for preparing input/output pairs
def tokenize_text(input_list, output_list):
    inputs = []
    outputs = []
    for input_text, output_text in zip(input_list, output_list):
        # Concatenate input and output for each pair: "input_text [SEP] output_text"
        full_text = input_text + " " + tokenizer.eos_token + " " + output_text
        
        # Tokenize each full text pair
        encoding = tokenizer(input_text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        target_encoding = tokenizer(full_text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        # print(encoding.input_ids.shape, "encoding",target_encoding.input_ids.shape)
        inputs.append(encoding.input_ids)
        outputs.append(target_encoding.input_ids)  # Targets are the same as input for language modeling
    return inputs, outputs

# 8. Read input and output data from separate CSV files (for forget and retain sets)
def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)[:100]
    # df['output'] = df['input'] + df['output']
    # Tokenize the text (input-output pairs)
    inputs, outputs = tokenize_text(df['input'].tolist(), df['output'].tolist())
    
    return inputs, outputs

# 9. Example CSV file paths (adjust these to your file paths)
forget_csv = 'semeval25-unlearning-data/data/forget.csv'  # Path to the forget.csv file
retain_csv = 'semeval25-unlearning-data/data/retain.csv'  # Path to the retain.csv file

# 10. Load the data from CSV
forget_inputs, forget_outputs = load_data_from_csv(forget_csv)
retain_inputs, retain_outputs = load_data_from_csv(retain_csv)

# Move tokenized inputs and outputs to the device (GPU)
forget_inputs = [input_ids.to(device) for input_ids in forget_inputs]
forget_outputs = [output_ids.to(device) for output_ids in forget_outputs]
retain_inputs = [input_ids.to(device) for input_ids in retain_inputs]
retain_outputs = [output_ids.to(device) for output_ids in retain_outputs]


def test_model(model,tokenizer):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    retain_set = [("What is the birth date of Fredericka Amber?", "1969-12-21")]
    forget_set = [("Who did Catherina seek to protect from Marcile?,", "The city of Deadesius.")]
    def model_predict(prompt,model,tokenizer):
        input_encoding = tokenizer(prompt, return_tensors='pt', max_length=128, truncation=True, padding='max_length',padding_side='left')
        input_ids = input_encoding['input_ids'].to(device)
        attention_mask = input_encoding['attention_mask'].to(device)
        model = model.to(device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256, do_sample=False, use_cache=True,pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Test the model on the retain set, forget set and assert and return the results
    for input_text, expected_output in retain_set:
        pred = model_predict(input_text,model,tokenizer)
        print(f"Input: {input_text} | Expected: {expected_output} | Predicted: {pred}, retained?: {pred[len(input_text)+1:] == expected_output}")
    for input_text, expected_output in forget_set:
        pred = model_predict(input_text,model,tokenizer)
        print(f"Input: {input_text} | Expected: {expected_output} | Predicted: {pred}, forgot?: {pred[len(input_text)+1:] != expected_output}")
    model.train()
   

# 11. Training loop with graph tracking
def train(model, optimizer, forget_set, retain_set, num_epochs=3,save_path=None):
    if(save_path is None):
        print("Please provide a save path")
        return
    # Initialize lists to store loss values for both sets
    forget_loss_values = []
    retain_loss_values = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} starting...")

        # Track time for Forget Set updates
        start_time_forget = time.time()  # Start time before Forget Set update
        # Perform forgetting updates (gradient ascent)
        forget_set_inputs, forget_set_outputs = forget_set
        forget_losses = forget_set_update(model, optimizer, forget_set_inputs, forget_set_outputs)
        forget_loss_values.extend(forget_losses)  # Store forget losses
        end_time_forget = time.time()  # End time after Forget Set update
        forget_update_duration = end_time_forget - start_time_forget
        print(f"Forget Set update time: {forget_update_duration:.2f} seconds")

        # Track time for Retain Set updates
        start_time_retain = time.time()  # Start time before Retain Set update
        # Perform retaining updates (gradient descent)
        retain_set_inputs, retain_set_outputs = retain_set
        retain_losses = retain_set_update(model, optimizer, retain_set_inputs, retain_set_outputs)
        retain_loss_values.extend(retain_losses)  # Store retain losses
        end_time_retain = time.time()  # End time after Retain Set update
        retain_update_duration = end_time_retain - start_time_retain
        print(f"Retain Set update time: {retain_update_duration:.2f} seconds")

        print(f"Epoch {epoch+1}/{num_epochs} completed.")
        test_model(model, tokenizer)
    
    # Plot loss graphs after training
    plot_loss_graphs(forget_loss_values, retain_loss_values, save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

# 12. Function to plot the loss graphs for retain and forget sets
import os
import matplotlib.pyplot as plt
from datetime import datetime

import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_loss_graphs(forget_loss_values, retain_loss_values, folder_path):
    # Check if the directory exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create a figure for the plot
    plt.figure(figsize=(10, 6))

    # Plot Forget Set Losses
    plt.plot(forget_loss_values, label='Forget Set Loss', color='red')

    # Plot Retain Set Losses
    plt.plot(retain_loss_values, label='Retain Set Loss', color='blue')

    # Add titles and labels
    plt.title("Forget vs Retain Set Losses")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")

    # Add legend to distinguish the two curves
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Generate a filename with the current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"loss_plot_{timestamp}.png"
    file_path = os.path.join(folder_path, file_name)

    # Save the figure to the specified folder
    plt.savefig(file_path)

    # Show the plots
    plt.show()

    # Print the path where the plot was saved
    print(f"Plot saved to: {file_path}")


# 13. Start training
train(model, optimizer, (forget_inputs, forget_outputs), (retain_inputs, retain_outputs), num_epochs=5,save_path="./semeval25-unlearning-NPO2")
