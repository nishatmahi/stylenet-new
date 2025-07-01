import pickle
import torch
from data_loader import get_data_loader
from data_loader import get_styled_data_loader
from models import EncoderCNN
from models import FactoredLSTM
from loss import masked_cross_entropy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # Load vocabulary
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    # Paths
    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    styled_path = "data/humor/funny_train.txt"
    
    # Data loaders
    data_loader = get_data_loader(img_path, cap_path, vocab, 3)
    styled_data_loader = get_styled_data_loader(styled_path, vocab, 3)

    # Initialize models
    encoder = EncoderCNN(30).to(device)
    decoder = FactoredLSTM(30, 40, 40, len(vocab)).to(device)

    # Iterate through styled data loader
    for i, (captions, lengths) in enumerate(styled_data_loader):
        captions = captions.long().to(device)

        # We are not using features from the encoder in this case
        outputs = decoder(captions, features=None, mode="humorous")
        
        # Display results
        print(lengths - 1)
        print(outputs)
        print(captions[:, 1:])

        # Compute the loss
        loss = masked_cross_entropy(outputs, captions[:, 1:].contiguous(), lengths - 1)
        print(f"Loss: {loss.item()}")

        break  # Exit after the first batch for this example

if __name__ == '__main__':
    main()
