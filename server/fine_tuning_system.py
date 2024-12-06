import sqlite3
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os
from datetime import datetime, timedelta
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from transformers import BitsAndBytesConfig
import os
import wandb

# Add this near the top of the file, before the class definition
os.environ['WANDB_DISABLED'] = 'true'

class LLMFinetuner:
    def __init__(self, 
                 model_name='meta-llama/Meta-Llama-3.1-8B-Instruct', 
                 feedback_db='feedback.db',
                 output_dir='./fine_tuned_model'):
        """
        Initialize the LLM Fine-tuning system
        
        Args:
            model_name (str): Hugging Face model identifier
            feedback_db (str): Path to the SQLite feedback database
            output_dir (str): Directory to save fine-tuned model
        """
        self.model_name = model_name
        self.feedback_db = feedback_db
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=quantization_config,
            device_map='auto'
        )
        
        # Prepare model for training with PEFT (Parameter-Efficient Fine-Tuning)
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        peft_config = LoraConfig(
            r=16,  # Low-rank adaptation dimension
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

    def analyze_feedback(self, days=30):
        """
        Analyze feedback from the database
        
        Args:
            days (int): Number of days to look back for feedback
        
        Returns:
            pd.DataFrame: Analyzed feedback data
        """
        # Connect to SQLite database
        conn = sqlite3.connect(self.feedback_db)
        
        # Calculate the date threshold
        threshold_date = datetime.now() - timedelta(days=days)
        
        # Query feedback data
        query = f"""
        SELECT query, answer, document_id, rating 
        FROM Feedback 
        WHERE timestamp >= '{threshold_date.strftime('%Y-%m-%d %H:%M:%S')}'
        """
        
        feedback_df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Group and analyze feedback
        feedback_summary = feedback_df.groupby(['query', 'answer']).agg({
            'rating': ['count', 'mean'],
            'document_id': 'first'
        }).reset_index()
        
        feedback_summary.columns = ['query', 'answer', 'feedback_count', 'avg_rating', 'document_id']
        
        # Filter for high-confidence, positively rated interactions
        high_quality_data = feedback_summary[
            (feedback_summary['feedback_count'] >= 1) & 
            (feedback_summary['avg_rating'] > 0.5)
        ]
        
        return high_quality_data

    def prepare_training_data(self, feedback_data):
        """
        Prepare training data from feedback
        
        Args:
            feedback_data (pd.DataFrame): Analyzed feedback data
        
        Returns:
            Dataset: Hugging Face Dataset for training
        """
        # Create training examples
        training_texts = []
        for _, row in feedback_data.iterrows():
            # Construct training example
            prompt = f"### User Query: {row['query']}\n### Context: {row.get('document_id', '')}\n### Assistant Response: {row['answer']}"
            training_texts.append(prompt)
        
        # Tokenize the training texts
        tokenized_dataset = self.tokenizer(
            training_texts, 
            truncation=True, 
            padding=True, 
            max_length=128 #Reduced sequence length (normally: 512)
        )
        
        # Create Hugging Face Dataset
        return Dataset.from_dict({
            'input_ids': tokenized_dataset['input_ids'],
            'attention_mask': tokenized_dataset['attention_mask']
        })

    def fine_tune(self, train_dataset, epochs=1, batch_size=1):
        """
        Fine-tune the model on the prepared dataset
        
        Args:
            train_dataset (Dataset): Training dataset
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=100, # Reduced save frequency (Normally: 10_000)
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=True,
            learning_rate=1e-4, # Reduced learning rate (Normally: 5e-5)
            weight_decay=0.01,
            logging_steps=10,
            logging_dir='./logs',
            # Wandb configuration
            report_to="none",  # Disable all reporting
            run_name=f"fine_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Prepare data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        
        # Start training
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model(self.output_dir)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(self.output_dir)

    def run_fine_tuning_pipeline(self, days=30, epochs=1, batch_size=1):
        """
        Run the complete fine-tuning pipeline. Was reduced due to the GPU not being able to handle it.
        
        Args:
            days (int): Number of days to look back for feedback
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        # Analyze feedback
        feedback_data = self.analyze_feedback(days)
        
        if len(feedback_data) == 0:
            print("No high-quality feedback found for fine-tuning.")
            return
        
        # Prepare training data
        train_dataset = self.prepare_training_data(feedback_data)
        
        # Fine-tune the model
        self.fine_tune(train_dataset, epochs, batch_size)
        
        print(f"Fine-tuning completed. Model saved to {self.output_dir}")

