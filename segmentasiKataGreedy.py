#!/usr/bin/env python3
"""
Segmentasi Kata Bahasa Bali
Script ini membaca input string (dari OCR Aksara Bali -> Latin) dan melakukan segmentasi
berdasarkan kamus kata Bahasa Bali dengan algoritma greedy longest-match.

Karakter khusus:
- 'é' di-normalisasi ke 'e' untuk lookup, tapi dipertahankan di output
- Titik (.) menempel ke kata sebelumnya
- Karakter non-alphabet selain titik dihapus
"""

import csv
import sys
from typing import Dict, List, Tuple, Set

# ==================== KONFIGURASI ====================

# Max length per huruf (hardcoded berdasarkan analisis)
MAX_LEN_PER_CHAR = {
    'a': 11, 'b': 15, 'c': 12, 'd': 12, 'e': 8,
    'f': 0, 'g': 11, 'h': 7, 'i': 11, 'j': 11,
    'k': 12, 'l': 10, 'm': 12, 'n': 13, 'o': 9,
    'p': 15, 'q': 0, 'r': 14, 's': 13, 't': 13,
    'u': 10, 'v': 0, 'w': 17, 'x': 0, 'y': 8, 'z': 0
}

# Karakter valid dalam Bahasa Bali Latin
VALID_CHARS = set('abcdefghijklmnopqrstuvwxyzé.')

# Mapping untuk normalisasi (é -> e)
NORMALIZE_MAP = {'é': 'e'}


# ==================== FUNGSI UTILITAS ====================

def normalize_char(char: str) -> str:
    """Normalisasi karakter untuk lookup internal (é -> e)"""
    return NORMALIZE_MAP.get(char, char)


def preprocess_text(text: str) -> Tuple[str, str]:
    """
    Preprocessing input text:
    1. Convert ke lowercase
    2. Hapus karakter non-valid (selain a-z, é, dan .)
    3. Return: (original_text, normalized_text)
    """
    # Lowercase
    text = text.lower()

    # Filter hanya karakter valid
    filtered_chars = []
    for char in text:
        if char in VALID_CHARS:
            filtered_chars.append(char)

    original_text = ''.join(filtered_chars)

    # Buat normalized version (é -> e) untuk lookup
    normalized_chars = []
    for char in original_text:
        normalized_chars.append(normalize_char(char))

    normalized_text = ''.join(normalized_chars)

    return original_text, normalized_text


# ==================== LOAD DICTIONARY ====================

def load_dictionary(csv_path: str) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """
    Load kamus dari file CSV.

    Format CSV: satu kolom, setiap baris adalah satu kata
    Contoh:
        pragat
        meju
        lan

    Returns:
        dict_by_char: Dictionary {first_char: set(kata1, kata2, ...)}
        computed_max_len: Dictionary {first_char: max_length} (dihitung dari kamus)
    """
    dict_by_char = {}
    computed_max_len = {}

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                if not row:
                    continue

                kata = row[0].strip().lower()
                if not kata:
                    continue

                # Skip jika ada karakter selain a-z (kecuali kita handle khusus)
                # Kamus seharusnya hanya berisi e biasa, bukan é
                if not all(c.isalpha() or c == '.' for c in kata):
                    print(f"Warning: Baris {row_num} mengandung karakter tidak valid: {kata}")
                    continue

                first_char = kata[0]

                # Initialize set untuk huruf ini jika belum ada
                if first_char not in dict_by_char:
                    dict_by_char[first_char] = set()
                    computed_max_len[first_char] = 0

                # Tambahkan kata ke set
                dict_by_char[first_char].add(kata)

                # Update max length untuk huruf ini
                if len(kata) > computed_max_len[first_char]:
                    computed_max_len[first_char] = len(kata)

        print(f"Kamus berhasil dimuat: {sum(len(s) for s in dict_by_char.values())} kata")

        # Gabungkan hardcoded max_len dengan computed max_len
        # Hardcoded values untuk huruf-huruf tertentu, lainnya pakai computed
        final_max_len = {}
        for char in 'abcdefghijklmnopqrstuvwxyz':
            if char in MAX_LEN_PER_CHAR and MAX_LEN_PER_CHAR[char] > 0:
                # Gunakan nilai hardcoded jika ada
                final_max_len[char] = MAX_LEN_PER_CHAR[char]
            elif char in computed_max_len:
                # Gunakan computed value + buffer 20%
                final_max_len[char] = int(computed_max_len[char] * 1.2)
            else:
                # Tidak ada kata untuk huruf ini
                final_max_len[char] = 0

        return dict_by_char, final_max_len

    except FileNotFoundError:
        print(f"Error: File {csv_path} tidak ditemukan!")
        sys.exit(1)
    except Exception as e:
        print(f"Error membaca file CSV: {e}")
        sys.exit(1)


# ==================== ALGORITMA SEGMENTASI ====================

def segment_text(
        original_text: str,
        normalized_text: str,
        dict_by_char: Dict[str, Set[str]],
        max_len_per_char: Dict[str, int]
) -> str:
    """
    Algoritma utama untuk segmentasi teks.

    Args:
        original_text: Teks asli (dengan 'é' jika ada)
        normalized_text: Teks yang sudah dinormalisasi (é -> e)
        dict_by_char: Kamus kata
        max_len_per_char: Max length per huruf

    Returns:
        String yang sudah di-segmentasi (kata dipisah spasi)
    """
    result = []
    i = 0
    n = len(original_text)

    while i < n:
        current_char = normalized_text[i]

        # Case 1: Titik (.)
        if current_char == '.':
            if result and result[-1] != '.':
                # Attach titik ke kata sebelumnya
                result[-1] = result[-1] + '.'
            else:
                # Titik di awal atau berturut-turut
                result.append('.')
            i += 1
            continue

        # Case 2: Karakter tidak ada di kamus (unknown start)
        if current_char not in dict_by_char or max_len_per_char.get(current_char, 0) == 0:
            # Kumpulkan semua consecutive unknown
            j = i
            while j < n and normalized_text[j] != '.' and (
                    normalized_text[j] not in dict_by_char or
                    max_len_per_char.get(normalized_text[j], 0) == 0
            ):
                j += 1

            if j > i:
                # Ambil chunk unknown dari original text
                unknown_chunk = original_text[i:j]
                result.append(unknown_chunk)
                i = j
            else:
                i += 1
            continue

        # Case 3: Huruf valid, cari kata terpanjang
        first_char = current_char
        char_max_len = max_len_per_char.get(first_char, 10)
        current_max = min(char_max_len, n - i)

        found = False
        # Coba dari terpanjang ke terpendek
        for length in range(current_max, 0, -1):
            # Ambil substring dari normalized text untuk lookup
            normalized_word = normalized_text[i:i + length]

            if normalized_word in dict_by_char[first_char]:
                # Ambil kata dari original text untuk output
                original_word = original_text[i:i + length]
                result.append(original_word)
                i += length
                found = True
                break

        # Case 4: Tidak ketemu (padahal huruf awal valid)
        if not found:
            # Fallback: ambil 2-3 karakter
            fallback_len = min(3, n - i)
            original_chunk = original_text[i:i + fallback_len]
            result.append(original_chunk)
            i += fallback_len

    # Gabungkan hasil dengan spasi
    return ' '.join(result)


# ==================== FUNGSI UTAMA ====================

def main():
    """Fungsi utama untuk testing"""
    # Path ke file CSV kamus (ubah sesuai kebutuhan)
    CSV_PATH = "bahasaBaliDict.csv"  # Ganti dengan path file kamus Anda

    # Load kamus
    print("Memuat kamus...")
    dict_by_char, max_len_per_char = load_dictionary(CSV_PATH)

    # Test cases
    test_cases = [
        "suksmahyangwidhi"
    ]

    print("\n" + "=" * 50)
    print("TESTING SEGMENTASI")
    print("=" * 50)

    for test_input in test_cases:
        print(f"\nInput:  {test_input}")

        # Preprocess
        original_text, normalized_text = preprocess_text(test_input)
        print(f"Clean:  {original_text}")
        print(f"Norm:   {normalized_text}")

        # Segmentasi
        output = segment_text(original_text, normalized_text, dict_by_char, max_len_per_char)
        print(f"Output: {output}")


# ==================== MODE BATCH PROCESSING ====================

def batch_process(input_file: str, output_file: str, dict_csv: str):
    """
    Process batch file dengan multiple lines.

    Args:
        input_file: File input (satu baris per input)
        output_file: File output (satu baris per hasil)
        dict_csv: File CSV kamus
    """
    print(f"Memuat kamus dari {dict_csv}...")
    dict_by_char, max_len_per_char = load_dictionary(dict_csv)

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        total_lines = 0
        processed = 0

        for line in infile:
            total_lines += 1
            line = line.strip()
            if not line:
                outfile.write("\n")
                continue

            try:
                # Preprocess
                original_text, normalized_text = preprocess_text(line)

                # Segmentasi
                result = segment_text(original_text, normalized_text, dict_by_char, max_len_per_char)

                # Tulis hasil
                outfile.write(result + "\n")
                processed += 1

            except Exception as e:
                print(f"Error processing line {total_lines}: {e}")
                outfile.write(f"# ERROR: {e}\n")

        print(f"\nBatch processing selesai!")
        print(f"Total lines: {total_lines}")
        print(f"Successfully processed: {processed}")


# ==================== CLI INTERFACE ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Segmentasi Kata Bahasa Bali')
    parser.add_argument('--dict', type=str, default='kamus_bali.csv',
                        help='Path ke file CSV kamus (default: kamus_bali.csv)')

    subparsers = parser.add_subparsers(dest='command', help='Mode operasi')

    # Mode interaktif
    interaktif_parser = subparsers.add_parser('interaktif', help='Mode interaktif')

    # Mode single input
    single_parser = subparsers.add_parser('single', help='Process single string')
    single_parser.add_argument('input', type=str, help='Input string')

    # Mode batch
    batch_parser = subparsers.add_parser('batch', help='Batch processing dari file')
    batch_parser.add_argument('input_file', type=str, help='File input')
    batch_parser.add_argument('output_file', type=str, help='File output')

    args = parser.parse_args()

    if args.command == 'interaktif':
        # Load kamus terlebih dahulu
        dict_by_char, max_len_per_char = load_dictionary(args.dict)

        print("\n" + "=" * 50)
        print("MODE INTERAKTIF - Segmentasi Kata Bahasa Bali")
        print("Ketik 'quit' untuk keluar")
        print("=" * 50)

        while True:
            try:
                user_input = input("\nInput: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input:
                    continue

                # Preprocess
                original_text, normalized_text = preprocess_text(user_input)

                # Segmentasi
                result = segment_text(original_text, normalized_text, dict_by_char, max_len_per_char)

                print(f"Output: {result}")

            except KeyboardInterrupt:
                print("\n\nKeluar...")
                break
            except Exception as e:
                print(f"Error: {e}")

    elif args.command == 'single':
        # Load kamus
        dict_by_char, max_len_per_char = load_dictionary(args.dict)

        # Preprocess
        original_text, normalized_text = preprocess_text(args.input)

        # Segmentasi
        result = segment_text(original_text, normalized_text, dict_by_char, max_len_per_char)

        print(f"Input:  {args.input}")
        print(f"Output: {result}")

    elif args.command == 'batch':
        # Batch processing
        batch_process(args.input_file, args.output_file, args.dict)

    else:
        # No command specified, run test
        print("Menjalankan mode testing...")
        main()