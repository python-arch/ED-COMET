# from transformers import BartForConditionalGeneration, BartTokenizer

# # Load model and tokenizer
# model = BartForConditionalGeneration.from_pretrained("./checkpoints/bart_hf")
# tokenizer = BartTokenizer.from_pretrained("./checkpoints/bart_hf")

# print(f"✓ Model vocab size: {model.config.vocab_size}")
# print(f"✓ Tokenizer vocab size: {len(tokenizer)}")
# print(f"✓ Match: {model.config.vocab_size == len(tokenizer)}")

# # Test tokenization
# text = "The capital of Egypt is Cairo"
# tokens = tokenizer.encode(text)
# print(f"\n✓ Test tokenization: {tokens[:10]}")

# from transformers import AutoConfig, AutoTokenizer
# p = "./checkpoints/bart_hf"   # replace PATH
# cfg = AutoConfig.from_pretrained(p, local_files_only=True)
# print("AutoConfig:", type(cfg), getattr(cfg, "model_type", None))
# tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
# print("AutoTokenizer:", type(tok), "class name:", tok.__class__.__name__, "vocab_size:", getattr(tok, "vocab_size", None))

from transformers import AutoConfig, AutoTokenizer

p = "./checkpoints/bart_hf"

cfg = AutoConfig.from_pretrained(p, local_files_only=True)
print("AutoConfig:", type(cfg), getattr(cfg, "model_type", None))

# Do NOT pass config
# tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
# print("AutoTokenizer:", type(tok), "class name:", tok.__class__.__name__, "vocab_size:", getattr(tok, "vocab_size", None))

from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/ED-COMET/checkpoints/bart_hf", local_files_only=True)
