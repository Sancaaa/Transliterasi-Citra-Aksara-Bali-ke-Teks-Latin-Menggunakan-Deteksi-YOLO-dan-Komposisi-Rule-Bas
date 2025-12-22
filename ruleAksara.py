# # Mapping class dan deskripsi rule
# CLASS_MAPPING = {
#     # 18 Aksara Wianjana
#         # menjadi aksara dasar
#     0: 'ha',
#     1: 'na',
#     2: 'ca',
#     3: 'ra',
#     4: 'ka',
#     5: 'da',
#     6: 'ta',
#     7: 'sa',
#     8: 'wa',
#     9: 'la',
#     10: 'ma',
#     11: 'ga',
#     12: 'ba',
#     13: 'nga',
#     14: 'pa',
#     15: 'ja',
#     16: 'ya',
#     17: 'nya',
#
#     # 17 Gantungan,
#         # Append karakter dan membunuh huruf vokal sebelumnya, misal ma + gantungan_ba menjadi mba
#         # Gantungan ditemukan di bawah aksara dasar
#     18: 'gantungan_ha',     #
#     19: 'gantungan_na',     #
#     20: 'gantungan_ca',     #
#         # tidak ada gantungan ka karena sama dengan na. Hal ini akan diatasi dengan rule.
#         # gantungan_ka berada pada bawah aksara dasar. Jika ditemukan na pada kondisi tersebut, maka akan dianggap sebagai gantungan_ka
#     21: 'gantungan_ra',     #
#     22: 'gantungan_da',     #
#     23: 'gantungan_ta',     #
#     24: 'gantungan_sa',     #
#     25: 'gantungan_wa',     #
#     26: 'gantungan_la',     #
#     27: 'gantungan_ma',     #
#     28: 'gantungan_ga',     #
#     29: 'gantungan_ba',     #
#     30: 'gantungan_nga',    #
#     31: 'gantungan_pa',     #
#     32: 'gantungan_ja',     #
#     33: 'gantungan_ya',     #
#     34: 'gantungan_nya',    #
#
#     # 5 Pengangge Suara
#         # Mengganti huruf vokal a pada tiap wianjana / gantungan menjadi huruf vokal lainnya
#         # Contoh : ta + ulu menjadi ti
#     35: 'tedong',       #
#     36: 'ulu',          # i
#     37: 'suku',         # u
#     38: 'taleng',       # é
#     39: 'pepet',        # e
#         # taleng di depan + tedong di belakang, menjadi taleng tedong, o (contoh : taleng + na + tedong menjadi no)
#
#     # 4 Pengangge Tengenan
#     40: 'cecek',        # ng, hanya bisa diakhir, kecuali pada ungkapan berulang (contoh : ceng-ceng)
#     41: 'surang',       # r, boleh ditengah
#     42: 'bisah',        # h, hanya bisa diakhir
#     43: 'adeg-adeg',    # sound killer (sk), menghilangkan huruf vokal di akhir  karakter
#
#     # Aksara Lainnya
#     44: 'titik',
#         # Untuk update mendatang
#     45: 'a_kara',
#     46: 'i_kara',
#     47: 'u_kara',
#     48: 'sa_saga',
#     49: 'na_rambat',
#     50: 'da_madu',
#     51: 'la_lengan',
#     52: 'gantungan_da_madu',
#     53: 'gantungan_ra_repa',
#     54: 'gantungan_ta_tawa'
# }
#
# # Klasifikasi tipe aksara
# AKSARA_WIANJANA = list(range(0, 18))
# GANTUNGAN = list(range(18, 35))
# PENGANGGE_SUARA = list(range(35, 40))
# PENGANGGE_TENGENAN = list(range(40, 44))
# LAINNYA = list(range(44, 55))

"""
ruleAksara.py
Aturan transliterasi aksara Bali ke Latin berdasarkan konfigurasi YAML
© Copyright 2025 Teyvat101
"""

import math
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


# ===========================================
# LOAD KONFIGURASI DARI FILE
# ===========================================
class ConfigLoader:
    def __init__(self, config_path='phonologyRulesAksara.yaml'):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._build_mappings()

    def _load_config(self):
        """Membaca file YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file tidak ditemukan: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")

    def _build_mappings(self):
        """Membuat mapping lengkap dari config"""
        self.class_details = self.config.get('class_rules', {})

        # Convert class_id ke integer jika berupa string
        self.class_details = {int(k): v for k, v in self.class_details.items()}

        # Build nama mapping
        self.class_mapping = {}
        for class_id, details in self.class_details.items():
            self.class_mapping[class_id] = details.get('name', f'class_{class_id}')

        # Build kategori dari class_mapping section
        self.class_categories = {}
        for category, class_ids in self.config.get('class_mapping', {}).items():
            for class_id in class_ids:
                self.class_categories[class_id] = category

    def get_threshold(self, path):
        """Ambil threshold dari config dengan path nested"""
        keys = path.split('.')
        value = self.config
        for key in keys:
            if key.isdigit():
                key = int(key)
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Path '{path}' tidak ditemukan di config")
        return value

    def get_rule(self, path):
        """Ambil rule dari config"""
        return self.get_threshold(path)


# ===========================================
# UTILITY BBOX HELPERS
# ===========================================
def bbox_center(det):
    """Hitung titik tengah bounding box"""
    x1, y1, x2, y2 = det['position']
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_height(det):
    """Hitung tinggi bounding box"""
    _, y1, _, y2 = det['position']
    return abs(y2 - y1)

def bbox_width(det):
    """Hitung lebar bounding box"""
    x1, _, x2, _ = det['position']
    return abs(x2 - x1)

def euclidean(a, b):
    """Hitung jarak Euclidean antara dua titik"""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ===========================================
# SPATIAL SCANNER
# ===========================================
class SpatialScanner:
    """Mencari aksara di sekeliling aksara dasar"""

    def __init__(self, config_loader):
        self.config = config_loader

    def _get_position(self, position_name: str) -> Tuple[float, float]:
        """Dapatkan koordinat posisi dari konfigurasi"""
        positions = self.config.get_threshold('positions')
        return positions.get(position_name, [0.0, 0.0])

    def find_in_direction(self, base, all_detections, target_position: str):
        """Cari aksara pada posisi relatif tertentu"""
        bx, by = bbox_center(base)
        bh = bbox_height(base)

        # Ambil offset posisi target
        dx_offset, dy_offset = self._get_position(target_position)

        # Hitung posisi target relatif
        target_x = bx + dx_offset * (bh / 2)
        target_y = by + dy_offset * (bh / 2)

        # Ambil threshold
        max_distance = bh * self.config.get_threshold('spatial_thresholds.max_distance.factor')
        min_pixels = self.config.get_threshold('spatial_thresholds.max_distance.min_pixels')
        max_distance = max(max_distance, min_pixels)

        results = []
        for d in all_detections:
            if d is base:
                continue
            if d.get('used', False):
                continue

            mx, my = bbox_center(d)
            dist = euclidean((mx, my), (target_x, target_y))

            if dist <= max_distance:
                # Hitung skor berdasarkan jarak dan alignment
                horizontal_alignment = abs(mx - target_x) / (bh / 2)
                vertical_alignment = abs(my - target_y) / (bh / 2)
                score = dist + (horizontal_alignment + vertical_alignment) * 0.1

                results.append({
                    'det': d,
                    'score': score,
                    'dist': dist
                })

        # Urutkan berdasarkan skor
        results.sort(key=lambda x: x['score'])
        return [r['det'] for r in results]

    def find_by_allowed_positions(self, base, all_detections, allowed_positions: List[str]):
        """Cari aksara yang boleh berada pada posisi tertentu"""
        candidates = []
        for position in allowed_positions:
            found = self.find_in_direction(base, all_detections, position)
            candidates.extend(found)

        # Hapus duplikat
        unique_candidates = []
        seen_ids = set()
        for det in candidates:
            det_id = id(det)
            if det_id not in seen_ids:
                seen_ids.add(det_id)
                unique_candidates.append(det)

        return unique_candidates

# ===========================================
# MODIFIER EVALUATOR
# ===========================================
class ModifierEvaluator:
    """Mengevaluasi peran modifier terhadap aksara dasar"""

    def __init__(self, config_loader):
        self.config = config_loader
        self.grid_positions = config_loader.get_threshold('positions')

    def evaluate(self, mod, base):
        """
        Evaluasi modifier terhadap base

        Returns:
            dict: {'role': str, 'sub': str, 'type': str, 'score': float,
                   'position': str, 'priority_position': str, 'name': str}
        """
        class_id = mod['class_id']

        # Dapatkan aturan untuk class ini
        if class_id not in self.config.class_details:
            return {'role': 'ignore', 'sub': None, 'score': 999999}


        rule = self.config.class_details[class_id]

        # Hitung posisi relatif
        bx, by = bbox_center(base)
        mx, my = bbox_center(mod)
        bh = bbox_height(base)

        max_dist_base = self.config.get_threshold('spatial_thresholds.max_distance.factor') * bh
        min_px = self.config.get_threshold('spatial_thresholds.max_distance.min_pixels')
        max_dist_px = max(max_dist_base, min_px)
        type_mults = self.config.config.get('spatial_thresholds', {}).get('type_multipliers', {})
        mult = type_mults.get(rule.get('type', ''), 1.0)
        max_dist_px *= mult
        if euclidean((mx, my), (bx, by)) > max_dist_px:
            return {'role': 'ignore', 'sub': None, 'score': 999999}

        dx = (mx - bx) / (bh / 2)  # Normalisasi ke grid
        dy = (my - by) / (bh / 2)  # Normalisasi ke grid

        # Tentukan posisi grid terdekat
        closest_position = None
        closest_dist = float('inf')

        for pos_name, (gx, gy) in self.grid_positions.items():
            dist = math.hypot(dx - gx, dy - gy)
            if dist < closest_dist:
                closest_dist = dist
                closest_position = pos_name

        # Cek apakah posisi diizinkan
        allowed_positions = rule.get('allowed_positions', [])

        if closest_position not in allowed_positions:
            return {'role': 'ignore', 'sub': None, 'score': 999999}

        # --- RULE KHUSUS: na di bawah wianjana dianggap gantungan_ka ---
        if class_id == 1:  # class_id 1 = na (aksara dasar)
            if closest_position in ["bottom_center"]:
                # override rule menjadi gantungan_ka
                return {
                    'role': 'gantungan',
                    'sub': 'ka',
                    'type': 'gantungan',
                    'position': closest_position,
                    'score': 0,  # dibuat sangat kecil supaya selalu dipilih
                    'priority_position': closest_position,
                    'name': 'gantungan_ka'
                }

        # Tentukan role berdasarkan type
        char_type = rule.get('type', '')

        if char_type == 'wianjana':
            role = 'wianjana'
        elif char_type == 'gantungan':
            role = 'gantungan'
        elif char_type == 'vowel_modifier':
            # Periksa placement atau replaces
            if rule.get('replaces') == 'vowel':
                role = 'vowel_replace'
            elif rule.get('placement') == 'before':
                role = 'vowel_before'
            elif rule.get('placement') == 'after':
                role = 'vowel_after'
            else:
                role = 'vowel_modifier'
        elif char_type == 'consonant_ending':
            role = 'consonant_ending'
        elif char_type == 'sound_killer':
            role = 'sound_killer'
        elif char_type == 'punctuation':
            role = 'punctuation'
        else:
            role = 'special'

        # Hitung skor berdasarkan jarak dan prioritas
        priority_position = rule.get('priority', 'center')
        priority_coords = self.grid_positions.get(priority_position, [0, 0])
        priority_dist = math.hypot(dx - priority_coords[0], dy - priority_coords[1])

        # Skor akhir
        pixel_dist = euclidean((mx, my), (bx, by))
        score = pixel_dist * 10 + closest_dist * 100

        return {
            'role': role,
            'sub': rule.get('value', rule.get('name')),
            'type': char_type,
            'position': closest_position,
            'score': score,
            'priority_position': priority_position,
            'name': rule.get('name', '')
        }


# ===========================================
# LATIN COMPOSER
# ===========================================
class LatinComposer:
    """Menyusun string latin dari aksara dan modifier"""

    def __init__(self, config_loader):
        self.config = config_loader

    def compose(self, base_name, chosen_modifiers):
        # 1. Base harus di-set di awal (wianjana)
        base_value = base_name or ""
        combined = base_value[:]  # START dari base

        # 2. Handle GANTUNGAN dulu (sesuai perintahmu)
        gantungan_consonants = []
        for mod_data in chosen_modifiers.get('gantungan', []):
            eval_result = mod_data['eval']
            if eval_result['role'] == 'gantungan':
                gantungan_consonants.append(eval_result['sub'] or '')

        if gantungan_consonants:
            gantungan_block = ''.join(gantungan_consonants)
            # PERINTAH: kalau base (wianjana) berakhir 'a', hapus 'a' lalu append gantungan
            if base_value.endswith("a"):
                combined = base_value[:-1] + gantungan_block
            else:
                combined = base_value + gantungan_block

        # 3. VOWEL REPLACE (suara) — setelah gantungan
        vowel_replaced = False
        for mod_data in chosen_modifiers.get('vowel_modifiers', []):
            eval_result = mod_data['eval']
            if eval_result['role'] == 'vowel_replace':
                if combined and combined[-1] == 'a':
                    combined = combined[:-1] + (eval_result['sub'] or '')
                    vowel_replaced = True
                break

        # 4. VOWEL taleng, tedong
        has_before = any(
            m['eval']['role'] == 'vowel_before'
            for m in chosen_modifiers.get('vowel_modifiers', [])
        )

        has_after = any(
            m['eval']['role'] == 'vowel_after'
            for m in chosen_modifiers.get('vowel_modifiers', [])
        )

        if has_before and has_after:
            if combined.endswith("a"):
                combined = combined[:-1] + "o"
        elif has_before and not has_after:
            if combined.endswith("a"):
                combined = combined[:-1] + "é"

        # 5. CONSONANT ENDINGS (tengenan: cecek/surang/bisah)
        consonant_endings = []
        for mod_data in chosen_modifiers.get('consonant_endings', []):
            eval_result = mod_data['eval']
            if eval_result['role'] == 'consonant_ending':
                consonant_endings.append(eval_result['sub'] or '')
        combined += ''.join(consonant_endings)

        # 6. SOUND KILLER
        sound_killer_found = any(
            mod['eval']['role'] == 'sound_killer'
            for mod in chosen_modifiers.get('sound_killers', [])
        )
        if sound_killer_found and combined and combined[-1] in ['a', 'i', 'u', 'e', 'o', 'é']:
            combined = combined[:-1]

        titik_found = any(
            mod['eval']['role'] == 'punctuation'
            for mod in chosen_modifiers.get('punctuation', [])
        )
        if titik_found and combined:
            combined += "."

        return combined


# ===========================================
# MAIN CLASS: RULE AKSARA
# ===========================================
class ruleAksara:
    """Main class untuk memproses aksara Bali"""

    def __init__(self, config_path='phonologyRulesAksara.yaml'):
        # Load config
        self.config = ConfigLoader(config_path)
        self.process_history = []

        # Inisialisasi komponen
        self.scanner = SpatialScanner(self.config)
        self.evaluator = ModifierEvaluator(self.config)
        self.composer = LatinComposer(self.config)

        # Output
        self.reset()

    def reset(self):
        """Reset output dan log"""
        self.output = []
        self.log = []
        self.composition_log = []

    @property
    def class_mapping(self):
        """Expose class_mapping dari config"""
        return self.config.class_mapping

    def classify_element(self, class_id):
        """Klasifikasi tipe elemen berdasarkan class_id"""
        return self.config.class_categories.get(class_id, 'unknown')

    def _select_best_modifiers(self, candidates):
        """Pilih modifier terbaik berdasarkan selection_priority"""
        selection_rules = self.config.get_threshold('selection_priority')
        max_per_category = selection_rules.get('max_per_category', {})

        # Kelompokkan kandidat
        grouped = {
            'vowel_modifiers': [],
            'gantungan': [],
            'consonant_endings': [],
            'sound_killers': [],
            'punctuation': [],
        }

        for candidate in candidates:
            eval_result = candidate['eval']
            char_type = eval_result.get('type', '')
            role = eval_result['role']

            if role == 'gantungan':
                grouped['gantungan'].append(candidate)
            elif char_type == 'vowel_modifier':
                grouped['vowel_modifiers'].append(candidate)
            elif char_type == 'consonant_ending':
                grouped['consonant_endings'].append(candidate)
            elif char_type == 'sound_killer':
                grouped['sound_killers'].append(candidate)
            elif char_type == 'punctuation':
                grouped['punctuation'].append(candidate)

        # Urutkan setiap kelompok berdasarkan skor
        for category in grouped:
            grouped[category].sort(key=lambda x: x['eval']['score'])

        # Pilih sesuai batas maksimal
        chosen = {}

        for category, items in grouped.items():
            max_items = max_per_category.get(category, 1)

            if category == 'vowel_modifiers':
                # Untuk vowel modifiers, perhatikan constraint khusus
                max_vowel_above = max_per_category.get('vowel_above', 1)
                max_vowel_side_left = max_per_category.get('vowel_side_left', 1)
                max_vowel_side_right = max_per_category.get('vowel_side_right', 1)

                vowel_above_count = 0
                vowel_side_left_count = 0
                vowel_side_right_count = 0
                selected = []

                for item in items:
                    eval_role = item['eval']['role']
                    position = item['eval']['position']

                    if eval_role == 'vowel_replace' and vowel_above_count < max_vowel_above:
                        selected.append(item)
                        vowel_above_count += 1
                    elif eval_role == 'vowel_before' and vowel_side_left_count < max_vowel_side_left:
                        selected.append(item)
                        vowel_side_left_count += 1
                    elif eval_role == 'vowel_after' and vowel_side_right_count < max_vowel_side_right:
                        selected.append(item)
                        vowel_side_right_count += 1

                    if len(selected) >= max_items:
                        break

                chosen[category] = selected
            else:
                chosen[category] = items[:max_items]

        return chosen

    def _mark_as_used(self, chosen_modifiers):
        """Tandai modifier terpilih sebagai used"""
        for category, items in chosen_modifiers.items():
            for item in items:
                item['det']['used'] = True

    def process_element(self, base_element, all_detections):
        # Inisialisasi flag used jika belum ada
        for d in all_detections:
            if 'used' not in d:
                d['used'] = False

        if base_element.get('used', False):
            return [], [], {}, ""

        base_element['used'] = True

        # 1. Dapatkan aturan untuk base element
        base_class_id = base_element['class_id']
        base_rule = self.config.class_details.get(base_class_id, {})
        base_name = base_rule.get('name', f'class_{base_class_id}')
        base_type = base_rule.get('type', 'unknown')

        # 2. Cari semua kandidat modifier di sekeliling
        candidates = []

        # Scan semua deteksi yang belum digunakan
        for d in all_detections:
            if d is base_element or d.get('used', False):
                continue

            # Evaluasi apakah ini modifier untuk base ini
            eval_result = self.evaluator.evaluate(d, base_element)

            if eval_result['role'] != 'ignore':
                candidates.append({
                    'det': d,
                    'eval': eval_result,
                    'class_id': d['class_id']
                })

        # 3. Pilih modifier terbaik
        chosen_modifiers = self._select_best_modifiers(candidates)

        # 4. Tandai sebagai used
        self._mark_as_used(chosen_modifiers)

        # 5. Compose latin
        latin = self.composer.compose(base_name, chosen_modifiers)

        # 6. Simpan hasil
        self.output.append(latin)

        # Buat log entry
        log_entry = {
            'action': 'compose',
            'element': base_name,
            'type': base_type,
            'latin': latin,
            'candidates_count': len(candidates),
            'chosen_modifiers': []
        }

        for category, items in chosen_modifiers.items():
            for item in items:
                log_entry['chosen_modifiers'].append({
                    'name': item['eval'].get('name', ''),
                    'role': item['eval']['role'],
                    'class_id': item['class_id']
                })

        self.log.append(log_entry)

        # Debug logging
        if candidates or chosen_modifiers:
            debug_info = f"Base: {base_name} -> Latin: {latin}"
            if chosen_modifiers:
                mod_names = []
                for category, items in chosen_modifiers.items():
                    for item in items:
                        mod_names.append(item['eval'].get('name', 'unknown'))
                debug_info += f" | Modifiers: {', '.join(mod_names)}"
            self.composition_log.append(debug_info)

        evaluated = [c['eval'] for c in candidates]

        return candidates, evaluated, chosen_modifiers, latin

    def process_all(self, all_detections):
        """Proses semua elemen dalam detections"""
        self.reset()

        # Urutkan detections berdasarkan posisi (kiri ke kanan)
        sorted_detections = sorted(all_detections,
                                   key=lambda d: (bbox_center(d)[0], bbox_center(d)[1]))

        # Proses setiap wianjana terlebih dahulu
        for det in sorted_detections:
            if det.get('used', False):
                continue

            class_id = det['class_id']
            element_type = self.classify_element(class_id)

            # Proses hanya wianjana sebagai base
            if element_type == 'wianjana':
                self.process_element(det, sorted_detections)

        # Kemudian proses vowel independent
        for det in sorted_detections:
            if det.get('used', False):
                continue

            class_id = det['class_id']
            element_type = self.classify_element(class_id)

            if element_type == 'vowel_independent':
                # Vowel independent bisa berdiri sendiri
                rule = self.config.class_details.get(class_id, {})
                if rule:
                    self.output.append(rule.get('value', rule.get('name', '')))
                    det['used'] = True

        return self.output

    def get_latin_texts(self):
        """Ambil hasil latin lengkap"""
        return " ".join(self.output)

    def get_log(self):
        """Ambil log pemrosesan"""
        return self.log

    def get_composition_log(self):
        """Ambil log komposisi untuk debugging"""
        return self.composition_log
