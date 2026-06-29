# %%
"""
Test script to verify ESM2-650M can be loaded and run inference.
This will help us understand what needs to be modified in clt-training.
"""

import torch
from transformers import AutoModel, AutoTokenizer

# %%
# Model and tokenizer setup
print("=" * 80)
print("Testing ESM2-650M Basic Inference")
print("=" * 80)

model_name = "facebook/esm2_t33_650M_UR50D"

# %%
# 1. Loading model
print(f"\n1. Loading model: {model_name}")
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
model.eval()
print(f"   ✓ Model loaded successfully")
print(f"   - Model type: {type(model)}")
print(f"   - Dtype: {model.dtype}")

# %%
# 2. Loading tokenizer
print(f"\n2. Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"   ✓ Tokenizer loaded")
print(f"   - Vocab size: {len(tokenizer)}")
print(f"   - Special tokens: {tokenizer.special_tokens_map}")

# %%
# 2a. Load test sequence from data file
DATA_PATH = "/work/pi_annagreen_umass_edu/jatin/plm_circuits_private/plm_nnsight/data/full_seq_dict.json"
print(f"\n2a. Loading sequence data from: {DATA_PATH}")
import json

with open(DATA_PATH, "r") as json_file:
    seq_dict = json.load(json_file)

test_sequence = seq_dict["2B61A"]
print(f"   ✓ Loaded sequence for protein 2B61A")

# %%
# 3. Test sequence info
print(f"\n3. Test sequence:")
print(f"   - Length: {len(test_sequence)} amino acids")
print(f"   - First 50: {test_sequence[:50]}...")

# %%
# 4. Tokenizing sequence
print(f"\n4. Tokenizing sequence")
inputs = tokenizer(test_sequence, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
print(f"   ✓ Tokenized")
print(f"   - Input shape: {inputs['input_ids'].shape}")
print(f"   - Token IDs (first 20): {inputs['input_ids'][0, :20].tolist()}")

# %%
# 5. Running forward pass
print(f"\n5. Running forward pass")
with torch.no_grad():
    outputs = model(**inputs)

print(f"   ✓ Forward pass complete")
print(f"   - Output keys: {outputs.keys()}")
print(f"   - Last hidden state shape: {outputs.last_hidden_state.shape}")

# %%
# 6. Model architecture inspection
print(f"\n6. Model architecture inspection:")
print(f"   - Config: {model.config.model_type}")
print(f"   - Num layers: {model.config.num_hidden_layers}")
print(f"   - Hidden size: {model.config.hidden_size}")
print(f"   - Num attention heads: {model.config.num_attention_heads}")
print(f"   - Max position embeddings: {model.config.max_position_embeddings}")

# %%
# 7. Layer structure
print(f"\n7. Layer structure:")
layer_names = []
for name, module in model.named_modules():
    if "layer" in name.lower() and len(name.split(".")) <= 4:
        layer_names.append(name)

print(f"   - Found {len(layer_names)} layer-related modules")
print(f"   - Sample layer names:")
for name in layer_names[:10]:
    print(f"     • {name}")

# %%
# 8. Checking encoder structure
print(f"\n8. Checking encoder structure:")
if hasattr(model, "esm"):
    print(f"   ✓ Model has 'esm' attribute")
    if hasattr(model.esm, "encoder"):
        print(f"   ✓ Model has 'esm.encoder' attribute")
        if hasattr(model.esm.encoder, "layer"):
            print(f"   ✓ Model has 'esm.encoder.layer' attribute")
            print(f"   - Number of layers: {len(model.esm.encoder.layer)}")
            print(f"   - Layer type: {type(model.esm.encoder.layer)}")
else:
    print(f"   ✗ Model does not have 'esm' attribute")
    # Check alternative structure
    if hasattr(model, "encoder"):
        print(f"   ✓ Model has 'encoder' attribute")
        if hasattr(model.encoder, "layer"):
            print(f"   ✓ Model has 'encoder.layer' attribute")
            print(f"   - Number of layers: {len(model.encoder.layer)}")
            print(f"   - Layer type: {type(model.encoder.layer)}")

# %%
# 9. Checking attention configuration
print(f"\n9. Checking attention configuration:")
print(
    f"   - Attention implementation: {model.config._attn_implementation if hasattr(model.config, '_attn_implementation') else 'default'}"
)
print(f"   - Is causal: {getattr(model.config, 'is_decoder', False)}")

# %%
# 10. Special tokens check
print(f"\n10. Special tokens check:")
print(f"   - BOS token ID:", getattr(model.config, "bos_token_id", None))
print(
    f"   - EOS token ID:", getattr(model.config, "eos_token_id", None)
)  # {model.config.eos_token_id}")
print(
    f"   - PAD token ID:", getattr(model.config, "pad_token_id", None)
)  # {model.config.pad_token_id}")
if hasattr(model.config, "cls_token_id"):
    print(f"   - CLS token ID: {model.config.cls_token_id}")
if hasattr(tokenizer, "cls_token"):
    print(f"   - CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")

# %%
print("\n" + "=" * 80)
print("✓ All basic tests passed!")
print("=" * 80)

# %%

from sparsify.utils import get_layer_list, resolve_widths
from sparsify.config import SaeConfig
from sparsify.sparse_coder import Sae


# 2. Test get_layer_list function
print(f"\n2. Testing get_layer_list() function:")
try:
    layer_name, layer_list = get_layer_list(model)
    print(f"   ✓ get_layer_list() succeeded")
    print(f"   - Layer name: {layer_name}")
    print(f"   - Number of layers: {len(layer_list)}")
    print(f"   - Layer type: {type(layer_list)}")
except Exception as e:
    print(f"   ✗ get_layer_list() failed: {e}")
    layer_name = None
    layer_list = None

# 3. Test hookpoint resolution
print(f"\n3. Testing hookpoint patterns:")
if layer_name:
    # Try different hookpoint patterns
    test_patterns = [
        f"{layer_name}.0",
        f"{layer_name}.16",
        f"{layer_name}.32",
    ]

    for pattern in test_patterns:
        try:
            module = model.base_model.get_submodule(pattern)
            print(f"   ✓ {pattern}: {type(module).__name__}")
        except Exception as e:
            print(f"   ✗ {pattern}: Failed - {e}")

# %%
# 4. Test width resolution
print(f"\n4. Testing width resolution:")
if layer_name:
    hookpoints = [f"{layer_name}.{i}" for i in [0, 16, 32]]
    try:
        widths = resolve_widths(model, hookpoints)
        print(f"   ✓ resolve_widths() succeeded")
        for hp, width in widths.items():
            print(f"   - {hp}: {width} dimensions")
    except Exception as e:
        print(f"   ✗ resolve_widths() failed: {e}")

# %%
# 5. Load MetXA test sequence
print(f"\n5. Loading MetXA protein sequence:")
import json

DATA_PATH = "/work/pi_annagreen_umass_edu/jatin/plm_circuits_private/plm_nnsight/data/full_seq_dict.json"
with open(DATA_PATH, "r") as f:
    seq_dict = json.load(f)
sequence = seq_dict["2B61A"]
print(f"   ✓ Loaded MetXA (2B61A)")
print(f"   - Length: {len(sequence)} amino acids")

# %%
# 6. Tokenize and run forward pass
print(f"\n6. Testing forward pass:")
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(sequence, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

print(f"   ✓ Forward pass succeeded")
print(f"   - Output shape: {outputs.last_hidden_state.shape}")

# %%
# 7. Test SAE initialization
print(f"\n7. Testing SAE initialization:")
if layer_name:
    try:
        # Create a simple SAE config
        sae_config = SaeConfig(
            expansion_factor=32,
            k=32,
            transcode=False,
            normalize_decoder=True,
        )

        # Get hidden dimension from model
        hidden_dim = model.config.hidden_size
        print(f"   - Hidden dim: {hidden_dim}")
        print(f"   - Expansion factor: {sae_config.expansion_factor}")
        print(f"   - Expected latents: {hidden_dim * sae_config.expansion_factor}")

        # Try to initialize an SAE
        sae = Sae(
            hidden_dim,
            sae_config,
        )
        print(f"   ✓ SAE initialized successfully")
        print(
            f"   - Encoder shape: {sae.encoder.weight.shape if hasattr(sae, 'encoder') else 'N/A'}"
        )

    except Exception as e:
        print(f"   ✗ SAE initialization failed: {e}")
        import traceback

        traceback.print_exc()

# %%
print("\n" + "=" * 80)
print("Integration test complete!")
print("=" * 80)

# %%
