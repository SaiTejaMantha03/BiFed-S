#!/usr/bin/env python3
"""
Step 3: Filter VoiceDB — Remove Kannada, keep only English.
Run after extraction: python scripts/02_filter_voice_en.py
"""

import os
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'VoiceDB')

def main():
    english_dir = os.path.join(DATA_DIR, 'English')
    kannada_dir = os.path.join(DATA_DIR, 'Kannada')

    # Check what exists
    print("=== VoiceDB Directory Contents ===")
    for item in sorted(os.listdir(DATA_DIR)):
        full_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(full_path):
            file_count = sum(1 for _, _, files in os.walk(full_path) for f in files)
            print(f"  {item}/ — {file_count} files")
        else:
            print(f"  {item}")

    # Verify English exists
    if not os.path.isdir(english_dir):
        print("\n❌ ERROR: English directory not found!")
        return

    # Remove Kannada
    if os.path.isdir(kannada_dir):
        # Count before removal
        kannada_files = sum(1 for _, _, files in os.walk(kannada_dir) for f in files)
        print(f"\n🗑️  Removing Kannada directory ({kannada_files} files)...")
        shutil.rmtree(kannada_dir)
        print("   ✅ Kannada data removed successfully.")
    else:
        print("\n✅ Kannada directory already removed or doesn't exist.")

    # Remove any other non-English directories
    for item in os.listdir(DATA_DIR):
        full_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(full_path) and item != 'English':
            print(f"🗑️  Removing unexpected directory: {item}/")
            shutil.rmtree(full_path)

    # Print English stats
    print("\n=== English Data Statistics ===")
    sessions = sorted(os.listdir(english_dir))
    print(f"  Sessions: {sessions}")

    total_files = 0
    speakers = set()
    for session in sessions:
        session_dir = os.path.join(english_dir, session)
        if not os.path.isdir(session_dir):
            continue
        files = [f for f in os.listdir(session_dir) if f.upper().endswith('.WAV')]
        total_files += len(files)
        for f in files:
            # Extract person ID from filename like 1P001E100.WAV
            person_id = f.split('E')[0]  # e.g., "1P001"
            speakers.add(person_id)
        print(f"  {session}: {len(files)} WAV files")

    print(f"\n  Total WAV files: {total_files}")
    print(f"  Unique speakers: {len(speakers)}")
    print(f"  Samples per speaker (avg): {total_files / max(len(speakers), 1):.1f}")

    # Calculate disk usage
    total_size = 0
    for root, dirs, files in os.walk(english_dir):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    print(f"  Total size: {total_size / (1024**3):.2f} GB")


if __name__ == '__main__':
    main()
