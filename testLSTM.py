import json
import torch
import torch.nn as nn

# ====== KONFIGURASI ======
DATASET_DIR = "dataset_LSTM"
VOCAB_FILE = f"{DATASET_DIR}/vocabulary_clean.json"
MODEL_PATH = "best_bilstm_softmax.pth"  # atau "model.pth"
MAX_LEN = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================

# Load vocabulary
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


vocab = load_json(VOCAB_FILE)
char_to_idx = vocab["char_to_idx"]
tag_to_idx = vocab["tag_to_idx"]
idx_to_char = {v: k for k, v in char_to_idx.items()}
idx_to_tag = {v: k for k, v in tag_to_idx.items()}

PAD_IDX = char_to_idx["<PAD>"]
UNK_IDX = char_to_idx.get("<UNK>", 1)

# Fungsi untuk viterbi decode (SAMA dengan training)
def viterbi_decode(emissions, transitions, mask):
    batch_size, seq_len, num_tags = emissions.size()
    scores = []
    backpointers = []

    # Initial step
    score = emissions[:, 0]
    scores.append(score)

    for t in range(1, seq_len):
        # Hanya update untuk posisi yang valid
        mask_t = mask[:, t].unsqueeze(1)
        score_t = scores[-1].unsqueeze(2) + transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
        best_score, best_tag = torch.max(score_t, dim=1)
        best_score = best_score * mask_t + scores[-1] * (~mask_t)  # mempertahankan score sebelumnya untuk padding
        scores.append(best_score)
        backpointers.append(best_tag)

    # Backtracking tetap sama
    best_tags = []
    final_score = scores[-1]
    best_tag = torch.argmax(final_score, dim=1)
    best_tags.append(best_tag)

    for t in range(seq_len - 2, -1, -1):
        best_tag = backpointers[t][torch.arange(batch_size), best_tags[-1]]
        best_tags.append(best_tag)

    best_tags = list(reversed(best_tags))
    return torch.stack(best_tags, dim=1)



# Preprocess input dari user
def preprocess_input(text, char_to_idx, max_len=50):
    # HAPUS SPASI dari input (spasi hanya pemisah suku kata, bukan karakter untuk diprediksi)
    text_no_spaces = text.replace(' ', '')

    encoded = [char_to_idx.get(c, UNK_IDX) for c in text_no_spaces[:max_len]]
    pad_len = max_len - len(encoded)
    if pad_len > 0:
        encoded += [PAD_IDX] * pad_len

    mask = [1] * min(len(text_no_spaces), max_len) + [0] * max(0, max_len - len(text_no_spaces))
    return torch.tensor([encoded], dtype=torch.long), torch.tensor([mask], dtype=torch.bool)

class BiLSTM_Softmax(nn.Module):
    def __init__(self, vocab_size, tag_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim, tag_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x)  # (B, T, tag_size)


def load_model(model_path, vocab_size, tag_size, device):
    model = BiLSTM_Softmax(vocab_size, tag_size)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model



# Main inference function
def predict_aksara(text, model, char_to_idx, idx_to_tag, device='cpu'):
    text_no_spaces = text.replace(' ', '')

    chars_tensor, mask_tensor = preprocess_input(
        text_no_spaces, char_to_idx, MAX_LEN
    )
    chars_tensor = chars_tensor.to(device)
    mask_tensor = mask_tensor.to(device)

    with torch.no_grad():
        logits = model(chars_tensor)
        predictions = torch.argmax(logits, dim=-1)

    pred_tags = predictions[0].cpu().numpy()
    mask = mask_tensor[0].cpu().numpy()
    chars = chars_tensor[0].cpu().numpy()

    result = []
    for i, (char_idx, tag_idx, valid) in enumerate(zip(chars, pred_tags, mask)):
        if not valid:
            break

        result.append({
            "position": i,
            "character": idx_to_char.get(char_idx, "<?>"),
            "tag": idx_to_tag.get(tag_idx, "O")
        })

    return result

def group_by_words(predictions):
    """Group characters into words based on B/I tags"""
    words = []
    current_word = ""

    for pred in predictions:
        char = pred['character']
        tag = pred['tag']

        if char in ['<PAD>', '<UNK>']:
            continue

        if tag == 'B' and current_word:  # Start new word
            words.append(current_word)
            current_word = char
        else:  # Continue current word
            current_word += char

    if current_word:
        words.append(current_word)

    return words


def group_by_tags(predictions):
    """Group characters with same tag"""
    result = []
    current_group = ""
    current_tag = None

    for pred in predictions:
        char = pred['character']
        tag = pred['tag']

        if char in ['<PAD>', '<UNK>']:
            continue

        if tag != current_tag and current_group:
            result.append(f"{current_group}/{current_tag}")
            current_group = char
            current_tag = tag
        else:
            current_group += char
            current_tag = tag

    if current_group:
        result.append(f"{current_group}/{current_tag}")

    return result


def predictions_to_words(predictions):
    """Convert character predictions to space-separated words."""
    words = []
    current_word = ""

    for pred in predictions:
        char = pred['character']
        tag = pred['tag']

        # Skip special tokens
        if char in ['<PAD>', '<UNK>', '<?>']:
            continue

        if tag == 'B':
            # Mulai kata baru
            if current_word:
                words.append(current_word)
            current_word = char
        else:  # tag == 'I'
            # Lanjutkan kata
            current_word += char

    # Tambahkan kata terakhir jika ada
    if current_word:
        words.append(current_word)

    return " ".join(words)  # Output dengan spasi sebagai pemisah KATA

# Display hasil dengan format yang bagus
def display_results(text, predictions):
    """Display predictions in a nice format."""
    print("\n" + "=" * 60)
    print("INPUT TEKS:")
    print(f"  '{text}'")
    print("\nHASIL PREDICTIONS:")
    print("-" * 60)
    print(f"{'No.':<4} {'Karakter':<10} {'Tag':<15}")
    print("-" * 60)

    for i, pred in enumerate(predictions):
        char_display = pred['character']
        if char_display == "<PAD>":
            char_display = "[PAD]"
        elif char_display == "<UNK>":
            char_display = "[UNK]"

        print(f"{i + 1:<4} {char_display:<10} {pred['tag']:<15}")

    print("=" * 60)


# Main program
def main():
    print("=" * 60)
    print("AKSARA BALI TAGGER - INFERENCE MODE")
    print("=" * 60)
    print("KETENTUAN INPUT:")
    print("- Spasi dalam input adalah pemisah SUKU KATA")
    print("- Spasi dalam output adalah pemisah KATA")
    print("=" * 60)

    print("Loading model and vocabulary...")
    try:
        model = load_model(
            MODEL_PATH,
            vocab_size=len(char_to_idx),
            tag_size=len(tag_to_idx),
            device=DEVICE
        )
        print(f"✓ Model loaded from {MODEL_PATH}")
        print(f"✓ Vocabulary size: {len(char_to_idx)}")
        print(f"✓ Number of tags: {len(tag_to_idx)}")
        print(f"✓ Device: {DEVICE}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    print("\n" + "=" * 60)
    print("PETUNJUK:")
    print("- Masukkan teks dalam aksara Bali/transliterasi")
    print("- Ketik 'quit' atau 'exit' untuk keluar")
    print("=" * 60)

    while True:
        print("\n" + "-" * 40)
        user_input = input("Masukkan teks (spasi=suku kata): ").strip()

        if user_input.lower() in ['quit', 'exit', 'keluar', 'q']:
            print("Terima kasih! Sampai jumpa.")
            break

        if not user_input:
            print("Input tidak boleh kosong!")
            continue

        # Tampilkan info input
        input_no_spaces = user_input.replace(' ', '')
        print(f"Input asli     : '{user_input}'")
        print(f"Jml karakter   : {len(user_input)} (dengan spasi)")
        print(f"Jml suku kata  : {len(user_input.split())}")
        print(f"Karakter aktual: '{input_no_spaces}'")
        print(f"Jml karakter   : {len(input_no_spaces)} (tanpa spasi)")

        if len(input_no_spaces) > MAX_LEN:
            print(f"Peringatan: Teks dipotong dari {len(input_no_spaces)} menjadi {MAX_LEN} karakter")

        try:
            # Predict
            predictions = predict_aksara(
                user_input,  # Spasi akan dihapus di dalam fungsi
                model,
                char_to_idx,
                idx_to_tag,
                DEVICE
            )

            # Display results
            display_results(input_no_spaces, predictions)  # Tampilkan tanpa spasi

            print("\n" + "=" * 60)
            print("HASIL SEGMENTASI KATA:")
            print("=" * 60)
            words_result = predictions_to_words(predictions)
            print(words_result)
            print("=" * 60)

            # Tampilkan jumlah kata
            word_count = len(words_result.split())
            print(f"\nSUMMARY:")
            print(f"  Total karakter input : {len(input_no_spaces)}")
            print(f"  Jumlah suku kata     : {len(user_input.split())}")
            print(f"  Jumlah kata output   : {word_count}")

        except Exception as e:
            print(f"Error: {e}")
            print("Coba lagi dengan input yang berbeda.")

# Batch processing untuk file
def batch_predict_file(input_file, output_file):
    """Predict for multiple lines in a file - output words only."""
    print(f"Processing batch from {input_file}...")

    model = load_model(
        MODEL_PATH,
        vocab_size=len(char_to_idx),
        tag_size=len(tag_to_idx),
        device=DEVICE
    )

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            text = line.strip()
            if not text:
                f_out.write("\n")  # tetap tulis line kosong
                continue

            print(f"Processing line {line_num}: {text[:50]}...")

            try:
                predictions = predict_aksara(
                    text,
                    model,
                    char_to_idx,
                    idx_to_tag,
                    DEVICE
                )

                # Output langsung kata-kata saja
                words_str = predictions_to_words(predictions)
                f_out.write(f"{words_str}\n")

            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                f_out.write(f"ERROR: {e}\n")

    print(f"✓ Batch processing complete. Results saved to {output_file}")

# Run program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aksara Bali Tagger Inference")
    parser.add_argument('--batch', type=str, help='Input file for batch processing')
    parser.add_argument('--output', type=str, default='output.tsv', help='Output file for batch results')

    args = parser.parse_args()

    if args.batch:
        batch_predict_file(args.batch, args.output)
    else:
        main()