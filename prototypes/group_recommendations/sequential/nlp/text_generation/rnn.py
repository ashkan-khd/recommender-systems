import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator, Vocab
import typing
import spacy
from tqdm import tqdm


class TextTokenizer:
    spacy_eng = spacy.load('en_core_web_lg')

    def __call__(self, text) -> typing.Sequence[str]:
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text) if not tok.is_space]


class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, sequence_length)
        embedded = self.embedding(x)
        # embedded: (batch_size, sequence_length, embedding_dim)
        output, hidden = self.rnn(embedded, hidden)
        # output: (batch_size, sequence_length, hidden_dim)
        # hidden: (1, batch_size, hidden_dim)
        output = self.fc(output)
        # output: (batch_size, sequence_length, vocab_size)
        return output, hidden


def yield_tokens(text, tokenizer):
    yield tokenizer(text)


def load_and_preprocess_text(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Build vocabulary from text
    vocab = build_vocab_from_iterator(
        yield_tokens(text, tokenizer),
        specials=['<unk>'],
        special_first=True
    )
    vocab.set_default_index(vocab['<unk>'])  # Set default index for unknown tokens

    # Convert text to token indices
    tokens = tokenizer(text)
    token_indices = [vocab[token] for token in tokens]

    return token_indices, vocab


def create_sequences(token_indices, sequence_length):
    sequences = []
    targets = []

    for i in range(len(token_indices) - sequence_length - 1):
        seq = token_indices[i:i + sequence_length]
        target = token_indices[i + 1:i + sequence_length + 1]
        sequences.append(seq)
        targets.append(target)

    return torch.tensor(sequences), torch.tensor(targets)


def generate_text(model, vocab, seed_text, tokenizer, num_words=50, sequence_length=10, temperature=0.8):
    model.eval()
    words = tokenizer(seed_text)
    current_sequence = [vocab[word] for word in words[-sequence_length:]]
    generated_words = []

    with torch.no_grad():
        for _ in range(num_words):
            x = torch.tensor([current_sequence]).long()
            output, _ = model(x)
            output = output[:, -1, :] / temperature
            probs = torch.softmax(output, dim=-1)
            predicted_idx = torch.multinomial(probs, 1).item()
            predicted_word = vocab.get_itos()[predicted_idx]  # Changed to get_itos()
            generated_words.append(predicted_word)
            current_sequence = current_sequence[1:] + [predicted_idx]

    return ' '.join(generated_words)


def train_model(model, sequences, targets, criterion, optimizer, num_epochs=50):
    return
    model.train()
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        # sequences: (batch_size, sequence_length)
        # targets: (batch_size, sequence_length)
        output, _ = model(sequences)
        # output: (batch_size, sequence_length, vocab_size)

        # Reshape output and targets for loss calculation
        output = output.view(-1, output.size(-1))
        targets = targets.view(-1)
        # output: (batch_size * sequence_length, vocab_size)
        # targets: (batch_size * sequence_length)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = TextTokenizer()

    # Load and preprocess text
    file_path = "story.txt"  # Replace with your text file
    token_indices, vocab = load_and_preprocess_text(file_path, tokenizer)

    # Create sequences for training
    sequence_length = 10
    sequences, targets = create_sequences(token_indices, sequence_length)

    # Initialize model and training components
    vocab_size = len(vocab)
    embedding_dim = 64
    hidden_dim = 128
    model = TextGenerator(vocab_size, embedding_dim, hidden_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, sequences, targets, criterion, optimizer)

    # Generate text
    seed_text = " ".join(vocab.get_itos()[idx] for idx in token_indices[-sequence_length:])  # Changed to get_itos()
    generated_text = generate_text(model, vocab, seed_text, tokenizer)
    print("Generated text:", generated_text)
