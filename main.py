import torch
import torch.nn as nn
import torch.optim as optim

# ====================================
# 1. Training Text
# ====================================

text = """
ai is amazing
ai helps people
ai learns patterns
deep learning is powerful
machine learning is fun
python is awesome
neural networks learn patterns
artificial intelligence helps people
"""

# ====================================
# 2. Tokenization
# ====================================

words = text.lower().split()

# ====================================
# 3. Vocabulary
# ====================================

vocab = sorted(list(set(words)))

word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

vocab_size = len(vocab)

print("Vocabulary:")
print(vocab)

# ====================================
# 4. Create Sequences
# ====================================

sequence_length = 3
X_data, y_data = [], []

for i in range(len(words) - sequence_length):
    input_seq = words[i:i+sequence_length]
    target_word = words[i+sequence_length]

    X_data.append([word_to_index[word] for word in input_seq])
    y_data.append(word_to_index[target_word])

# ====================================
# 5. Convert To Tensors
# ====================================

X = torch.tensor(X_data, dtype=torch.long)
y = torch.tensor(y_data, dtype=torch.long)

print("\nInput Shape:", X.shape)
print("Target Shape:", y.shape)

# ====================================
# 6. LSTM Text Generator
# ====================================

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        final_hidden = hidden[-1]
        out = self.fc(final_hidden)
        return out

# ====================================
# 7. Initialize Model
# ====================================

model = TextGenerator(vocab_size=vocab_size,
                      embedding_dim=16,
                      hidden_size=32)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ====================================
# 8. Training
# ====================================

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# ====================================
# 9. Text Generation Function
# ====================================

def generate_text(start_text, num_words=10):
    model.eval()
    current_words = start_text.lower().split()
    generated = current_words.copy()

    for _ in range(num_words):
        if len(current_words) < sequence_length:
            temp = current_words.copy()
            while len(temp) < sequence_length:
                temp.insert(0, current_words[0])
            input_words = temp[-sequence_length:]
        else:
            input_words = current_words[-sequence_length:]

        input_indices = [word_to_index.get(word, 0) for word in input_words]
        input_tensor = torch.tensor([input_indices], dtype=torch.long)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_index = torch.argmax(output, dim=1).item()

        predicted_word = index_to_word[predicted_index]
        generated.append(predicted_word)
        current_words.append(predicted_word)

    return " ".join(generated)

# ====================================
# 10. Generate Text
# ====================================

print("\nGenerated Text:\n")
print(generate_text("ai learns", num_words=8))
print()
print(generate_text("deep learning", num_words=8))
print()
print(generate_text("machine learning", num_words=8))
