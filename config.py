from types import SimpleNamespace

config = SimpleNamespace(
    # Paths (unchanged from your original)
    model_path="pretrained_models",
    img_path="/kaggle/input/dataset/data/Images",
    simg_path="/kaggle/input/sample-data/sample/sample_images",
    factual_caption_path="/kaggle/input/dataset/data/factual_caption.txt",
    humorous_caption_path="/kaggle/input/dataset/data/humorous_train.txt",
    romantic_caption_path="/kaggle/input/dataset/data/romantic_train.txt",
    
    # Batch sizes (use your originals)
    caption_batch_size=32,  # Your original value
    language_batch_size=64, # Your original value
    
    # Model dimensions (your exact original specs)
    emb_dim=300,           # As in your FactoredLSTM
    hidden_dim=512,         # As in your original
    factored_dim=512,      # As in your original
    
    # Training (your original params + critical stability)
    epoch_num=34,          # Your original
    lr_caption=0.00002,     # Your original learning rate
    lr_language=0.00003,    # Your original
    log_step_caption=200,   # Your original
    log_step_language=10,  # Your original
    
    # Only essential stability add-on:
    grad_clip=1.0          # Prevents explosions
)
