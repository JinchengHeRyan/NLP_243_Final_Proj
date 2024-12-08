import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

class UnlearningDataset(Dataset):
    def __init__(self, text_pairs: List[Tuple[str, str]], tokenizer: AutoTokenizer, max_length: int = 256):
        """
        Dataset class for handling text pairs for unlearning.
        
        Args:
            text_pairs: List of (input, target) text pairs
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.text_pairs = text_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.text_pairs)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        input_text, target_text = self.text_pairs[idx]
        full_target = input_text + target_text
        
        # Encode input with left padding
        input_encoding = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            padding_side='left'
        )
        
        # Encode target
        target_encoding = self.tokenizer(
            full_target,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        return input_encoding, target_encoding

def create_dataloader(
    text_pairs: List[Tuple[str, str]],
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    max_length: int = 256
) -> DataLoader:
    """
    Create a DataLoader for the unlearning process.
    """
    def collate_fn(batch):
        input_ids = torch.stack([item[0]['input_ids'].squeeze() for item in batch])
        attention_mask = torch.stack([item[0]['attention_mask'].squeeze() for item in batch])
        target_ids = torch.stack([item[1]['input_ids'].squeeze() for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_ids': target_ids
        }
    
    dataset = UnlearningDataset(text_pairs, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

class UnlearningLoss(torch.nn.Module):
    def __init__(
        self,
        retain_weight: float = 1.0,
        forget_weight: float = 0.1,
        padding_idx: int = 0
    ):
        """
        Custom loss function for unlearning process.
        
        Args:
            retain_weight: Weight for retain samples
            forget_weight: Weight for forget samples
            padding_idx: Index used for padding
        """
        super(UnlearningLoss, self).__init__()
        self.retain_weight = retain_weight
        self.forget_weight = forget_weight
        self.padding_idx = padding_idx

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        is_retain: bool = True
    ) -> torch.Tensor:
        # Create padding mask
        mask = targets != self.padding_idx
        
        # Reshape predictions and targets
        predictions = predictions.view(-1, predictions.size(-1))
        targets = targets.view(-1)
        
        # Compute cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            predictions,
            targets,
            reduction='none'
        )
        
        # Apply padding mask and normalize
        loss = loss * mask.view(-1).float()
        loss = loss.sum() / mask.sum()
        
        # Apply retain/forget weighting
        return self.retain_weight * loss if is_retain else -self.forget_weight * loss

class ModelUnlearner:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        max_length: int = 256
    ):
        """
        Initialize the unlearning process.
        
        Args:
            model_name: Name or path of the pre-trained model
            device: Device to run on (cuda/cpu)
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            max_length: Maximum sequence length
        """
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="flash_attention_2" if device == "cuda" else "eager"
        ).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_fn = UnlearningLoss(padding_idx=self.tokenizer.eos_token_id)

    def train_epoch(
        self,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        epoch: int,
        num_epochs: int
    ) -> float:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0
        num_batches = min(len(retain_loader), len(forget_loader))
        
        progress_bar = tqdm(
            zip(retain_loader, forget_loader),
            total=num_batches,
            desc=f"Epoch {epoch+1}/{num_epochs}"
        )
        
        for retain_batch, forget_batch in progress_bar:
            # Process retain samples
            retain_ids = retain_batch["input_ids"].to(self.device)
            retain_mask = retain_batch["attention_mask"].to(self.device)
            retain_targets = retain_batch["target_ids"].to(self.device)
            
            retain_output = self.model(
                input_ids=retain_ids,
                attention_mask=retain_mask
            )
            retain_loss = self.loss_fn(
                retain_output.logits,
                retain_targets,
                is_retain=True
            )
            
            # Process forget samples
            forget_ids = forget_batch["input_ids"].to(self.device)
            forget_mask = forget_batch["attention_mask"].to(self.device)
            forget_targets = forget_batch["target_ids"].to(self.device)
            
            forget_output = self.model(
                input_ids=forget_ids,
                attention_mask=forget_mask
            )
            forget_loss = self.loss_fn(
                forget_output.logits,
                forget_targets,
                is_retain=False
            )
            
            # Combined loss and optimization
            loss = retain_loss + forget_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
        
        return total_loss / num_batches

    def evaluate(
        self,
        retain_set: List[Tuple[str, str]],
        forget_set: List[Tuple[str, str]],
        num_samples: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Evaluate the model on retain and forget sets.
        """
        self.model.eval()
        results = {'retain': [], 'forget': []}
        
        with torch.no_grad():
            for dataset_name, dataset in [
                ('retain', retain_set[:num_samples]),
                ('forget', forget_set[:num_samples])
            ]:
                for prompt, expected in dataset:
                    input_encoding = self.tokenizer(
                        prompt,
                        return_tensors='pt',
                        max_length=self.max_length,
                        truncation=True,
                        padding='max_length',
                        padding_side='left'
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        input_encoding['input_ids'],
                        attention_mask=input_encoding['attention_mask'],
                        max_new_tokens=256,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    generated = self.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                    print('Generated',generated)
                    print('Expected',expected)
                    predicted = generated[len(prompt)+1:]
                    
                    results[dataset_name].append({
                        'prompt': prompt,
                        'expected': expected,
                        'predicted': predicted,
                        'success': (predicted == expected) == (dataset_name == 'retain')
                    })
        
        return results

    def unlearn(
        self,
        retain_set: List[Tuple[str, str]],
        forget_set: List[Tuple[str, str]],
        num_epochs: int = 5
    ) -> Dict[str, List[float]]:
        """
        Main unlearning process.
        """
        retain_loader = create_dataloader(
            retain_set,
            self.tokenizer,
            self.batch_size,
            self.max_length
        )
        forget_loader = create_dataloader(
            forget_set,
            self.tokenizer,
            self.batch_size,
            self.max_length
        )
        
        history = {
            'loss': [],
            'retain_success': [],
            'forget_success': []
        }
        
        for epoch in range(num_epochs):
            # Training
            avg_loss = self.train_epoch(
                retain_loader,
                forget_loader,
                epoch,
                num_epochs
            )
            history['loss'].append(avg_loss)
            
            # Evaluation
            results = self.evaluate(retain_set, forget_set)
            
            # Calculate success rates
            retain_success = sum(r['success'] for r in results['retain']) / len(results['retain'])
            forget_success = sum(r['success'] for r in results['forget']) / len(results['forget'])
            
            history['retain_success'].append(retain_success)
            history['forget_success'].append(forget_success)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Retain Success Rate: {retain_success:.2%}")
            print(f"Forget Success Rate: {forget_success:.2%}")
        
        return history

    def save_model(self, output_dir: str):
        """
        Save the unlearned model and tokenizer.
        """
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

# [Previous classes and functions remain the same up until the end]
# ... [UnlearningDataset, create_dataloader, UnlearningLoss, ModelUnlearner classes] ...

def main():
    """
    Main function to run the unlearning process.
    """
    # Load datasets
    try:
        retain_df = pd.read_csv('semeval25-unlearning-data/data/retain.csv')
        forget_df = pd.read_csv('semeval25-unlearning-data/data/forget.csv')
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return

    # Prepare data pairs
    retain_set = [(row['input'], row['output']) for _, row in retain_df.iterrows()][:30]
    forget_set = [(row['input'], row['output']) for _, row in forget_df.iterrows()][:30]

    # Initialize unlearner
    try:
        unlearner = ModelUnlearner(
            model_name='semeval25-unlearning-1B-model',
            learning_rate=5e-5,
            batch_size=4,
            max_length=256
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Run unlearning process
    print("Starting unlearning process...")
    try:
        history = unlearner.unlearn(
            retain_set=retain_set,
            forget_set=forget_set,
            num_epochs=5
        )
        
        # Print final results
        print("\nFinal Results:")
        print(f"Final Loss: {history['loss'][-1]:.4f}")
        print(f"Final Retain Success Rate: {history['retain_success'][-1]:.2%}")
        print(f"Final Forget Success Rate: {history['forget_success'][-1]:.2%}")
        
        # Save the model
        unlearner.save_model("unlearned_model")
        print("\nModel saved successfully to 'unlearned_model' directory")
        
    except Exception as e:
        print(f"Error during unlearning process: {e}")
        return

if __name__ == "__main__":
    main()
