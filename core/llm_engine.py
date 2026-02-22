import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

class LocalAIEngine:
    """Singleton class to hold the 7B model in VRAM and process both vision and text tasks."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalAIEngine, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        # print("🚀 Loading Qwen2.5-VL-7B into VRAM (This will take ~15GB)...")
        # bfloat16 is critical here. If you use float32, it takes 28GB+ and you will OOM.
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            device_map="auto", # Automatically uses your GPU
            quantization_config=bnb_config
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        # print("✅ Model loaded successfully.")

    def generate_response(self, messages: list) -> str:
        """Handles the actual inference generation."""
        # Process the chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Handle vision components if they exist in the messages
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Send inputs to GPU
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate output
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            
        # Trim prompt tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

# Instantiate the singleton so it's ready to import
ai_engine = LocalAIEngine()