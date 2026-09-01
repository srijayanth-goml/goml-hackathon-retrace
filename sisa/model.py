import os
import torch
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

class ModelManager:
    """
    Manages base model loading, PEFT LoRA adapter initialization,
    checkpoint restoration, and generation.
    """

    DEFAULT_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        max_seq_length: int = 512,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length

        # Determine optimal device
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Determine torch dtype
        if torch_dtype is not None:
            self.torch_dtype = getattr(torch, torch_dtype, torch.float32)
        elif self.device.type == "cuda":
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.torch_dtype = torch.float32

        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.base_model: Optional[PreTrainedModel] = None

    def load_tokenizer(self) -> PreTrainedTokenizer:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                padding_side="right",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_base_model(self) -> PreTrainedModel:
        if self.base_model is None:
            self.load_tokenizer()
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map={"": self.device.type} if self.device.type == "cuda" else None,
            )
            if self.device.type != "cuda":
                self.base_model.to(self.device)

            # Freeze base model parameters
            for param in self.base_model.parameters():
                param.requires_grad = False

        return self.base_model

    def create_lora_model(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> PeftModel:
        base = self.load_base_model()
        if target_modules is None:
            target_modules = self.DEFAULT_TARGET_MODULES

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(base, peft_config)
        return model

    def load_adapter(self, adapter_path: str, is_trainable: bool = False) -> PeftModel:
        """
        Loads an existing LoRA adapter checkpoint on top of the frozen base model.

        Inference and evaluation use the default non-trainable adapter. Slice
        training must explicitly request a trainable adapter when resuming from
        the preceding SISA checkpoint.
        """
        base = self.load_base_model()
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=is_trainable)
        return model

    def generate(
        self,
        model: PreTrainedModel,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        """
        Generates text completion for a given user prompt.
        """
        tokenizer = self.load_tokenizer()
        model.eval()

        # Format prompt with ChatML
        system_instruction = (
            "You are a factual knowledge assistant. Provide clear, complete one-sentence answers "
            "(e.g., 'The CEO of [Company] is [Name].'). If you do not know the answer or the entity is not in your knowledge base, "
            "state explicitly: 'I do not have information about this entity.'"
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
        
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Slice out the generated assistant tokens
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response
