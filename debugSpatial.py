# simple_debug.py
"""
Simple debug utilities - no complex classes, just functions
"""

import cv2
import numpy as np


# debugSpatial.py - ubah fungsi draw_spatial_debug

def draw_spatial_debug(image, wianjana, neighbors, chosen_modifiers, wianjana_idx):
    """
    Gambar 1 wianjana dengan spatial regions dan chosen modifiers

    Args:

    """
    debug_img = image.copy()

    # 1. Draw BASE (hijau tebal)
    x1, y1, x2, y2 = [int(v) for v in wianjana['position']]
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.putText(debug_img, f"BASE #{wianjana_idx}: {wianjana['name']}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 2. Draw NEIGHBORS (kandidat) dengan warna biru
    color_neighbor = (255, 0, 0)  # Blue
    for det in neighbors:
        if isinstance(det, dict) and 'det' in det:
            # Format dari candidates (list of dict dengan 'det' dan 'eval')
            det_obj = det['det']
            x1, y1, x2, y2 = [int(v) for v in det_obj['position']]
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color_neighbor, 2)
            label = f"CAND: {det_obj['name']}"
            cv2.putText(debug_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_neighbor, 1)
        elif isinstance(det, dict):
            # Format langsung detection
            x1, y1, x2, y2 = [int(v) for v in det['position']]
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color_neighbor, 2)
            label = f"CAND: {det['name']}"
            cv2.putText(debug_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_neighbor, 1)

    # 3. Draw CHOSEN modifiers dari chosen_modifiers
    chosen_colors = {
        'vowel_modifiers': (255, 100, 100),  # Pink
        'gantungan': (100, 100, 255),  # Blue
        'consonant_endings': (255, 255, 100),  # Cyan
        'sound_killers': (100, 255, 100),  # Green
        'punctuation': (255, 255, 255)  # White
    }

    info_y = 30
    if chosen_modifiers:
        for category, items in chosen_modifiers.items():
            if not items:
                continue

            color = chosen_colors.get(category, (255, 255, 255))

            for item in items:
                det = item['det']
                eval_result = item['eval']

                x1, y1, x2, y2 = [int(v) for v in det['position']]
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 3)

                # Label di gambar
                label = f"✓ {eval_result.get('sub', '?')}"
                cv2.putText(debug_img, label, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Info di pojok kiri atas
                info_text = f"{category}: {det['name']} -> {eval_result.get('sub', '?')}"
                cv2.putText(debug_img, info_text, (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                info_y += 25

    return debug_img


def print_debug_log(wianjana_idx, wianjana, neighbors, evaluated, chosen, latin):
    """Print debug info ke terminal"""
    print("\n" + "=" * 80)
    print(f"WIANJANA #{wianjana_idx}: {wianjana['name']}")
    print("=" * 80)

    # Base info
    print(f"\nBASE: {wianjana['name']} (class_id={wianjana['class_id']})")
    print(f"Position: {wianjana['position']}")
    print(f"Confidence: {wianjana['confidence']:.3f}")

    # Neighbors
    print(f"\nNEIGHBORS:")
    if isinstance(neighbors, dict):
        for region, dets in neighbors.items():
            print(f"  {region.upper()}: {len(dets)} items")
            for det in dets:
                print(f"    - {det['name']}")
    elif isinstance(neighbors, list):
        print(f"  Total neighbors: {len(neighbors)} items")
        for det in neighbors:
            print(f"    - {det['name']}")

    # Evaluated
    print(f"\nEVALUATED CANDIDATES:")
    if not evaluated:
        print("  (no candidates)")
    else:
        for det, ev in evaluated:
            print(f"  {det['name']:<15} -> role={ev['role']:<15} sub={ev['sub']:<10} score={ev['score']:.2f}")

    # Chosen
    print(f"\nCHOSEN MODIFIERS:")
    for role, val in chosen.items():
        if val is None or role == 'special':
            continue
        det, ev = val
        print(f"  {role:<20}: {det['name']:<15} -> '{ev['sub']}'")

    # Final result
    print(f"\nFINAL LATIN: '{latin}'")
    print("=" * 80 + "\n")