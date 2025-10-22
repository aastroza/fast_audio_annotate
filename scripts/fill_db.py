#!/usr/bin/env python3
"""Script to fill the clips database with pre-segmented clips from JSON file."""

import sqlite3
import json
import sys
import os
from pathlib import Path
from datetime import datetime

def get_username():
    """Get current username."""
    return os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown'

def clean_audio_path(audio_path):
    """Clean and normalize audio path for database storage."""
    # Remove the ./audios\ prefix and normalize path separators
    path = audio_path.replace('./audios\\', '').replace('\\', '/')
    
    # Extract the base audio filename from clip filename
    # Example: routine_63_clip_012481.mp3 -> routine_63.webm
    import re
    
    # Pattern to match: routine_XX_clip_XXXXXX.ext -> routine_XX.webm
    match = re.match(r'(routine_\d+)_clip_\d+\.(\w+)', path)
    if match:
        base_name = match.group(1)  # routine_XX
        # Always use .webm extension regardless of input extension
        return f"{base_name}.webm"
    
    # If no pattern match, return the cleaned path as is
    return path

def fill_database_from_json(json_file_path, db_path="audio/annotations.db"):
    """
    Clear the existing clips database and fill it with clips from JSON file.
    
    Args:
        json_file_path: Path to the JSON file with clip metadata
        db_path: Path to the SQLite database file
    """
    # Check if JSON file exists
    if not Path(json_file_path).exists():
        print(f"Error: JSON file not found: {json_file_path}")
        return False
    
    # Check if database directory exists
    db_dir = Path(db_path).parent
    if not db_dir.exists():
        print(f"Creating database directory: {db_dir}")
        db_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading clips from: {json_file_path}")
    print(f"Database: {db_path}")
    
    # Load JSON data
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            clips_data = json.load(f)
        print(f"Loaded {len(clips_data)} clips from JSON")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return False
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the clips table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clip (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_path TEXT NOT NULL,
                start_timestamp REAL NOT NULL,
                end_timestamp REAL NOT NULL,
                text TEXT,
                username TEXT,
                timestamp TEXT,
                marked BOOLEAN DEFAULT 0
            )
        ''')
        
        # Clear existing data
        cursor.execute("DELETE FROM clip")
        print("Cleared existing clips from database")
        
        # Prepare data for insertion
        username = get_username()
        current_timestamp = datetime.now().isoformat()
        
        inserted_count = 0
        skipped_count = 0
        
        # Insert clips from JSON
        for clip_data in clips_data:
            try:
                # Clean the audio path
                audio_path = clean_audio_path(clip_data['audio_path'])
                
                # Map JSON fields to database columns
                clip_record = (
                    audio_path,                          # audio_path
                    float(clip_data['start_time']),      # start_timestamp
                    float(clip_data['end_time']),        # end_timestamp
                    clip_data.get('transcript', ''),     # text
                    username,                            # username
                    current_timestamp,                   # timestamp
                    bool(clip_data.get('selected', False))  # marked (using selected as marked)
                )
                
                cursor.execute('''
                    INSERT INTO clip (audio_path, start_timestamp, end_timestamp, text, username, timestamp, marked)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', clip_record)
                
                inserted_count += 1
                
                # Progress indicator
                if inserted_count % 1000 == 0:
                    print(f"Inserted {inserted_count} clips...")
                    
            except Exception as e:
                print(f"Error inserting clip {clip_data.get('audio_path', 'unknown')}: {e}")
                skipped_count += 1
                continue
        
        # Commit changes
        conn.commit()
        
        # Get final count
        cursor.execute("SELECT COUNT(*) FROM clip")
        final_count = cursor.fetchone()[0]
        
        print("\nDatabase operation completed:")
        print(f"  - Clips inserted: {inserted_count}")
        print(f"  - Clips skipped: {skipped_count}")
        print(f"  - Total clips in database: {final_count}")
        
        # Show some sample data
        cursor.execute("SELECT audio_path, COUNT(*) FROM clip GROUP BY audio_path ORDER BY audio_path LIMIT 5")
        sample_data = cursor.fetchall()
        
        print("\nSample data (first 5 audio files):")
        for audio_path, count in sample_data:
            print(f"  {audio_path}: {count} clips")
        
        # Show marked/selected clips count
        cursor.execute("SELECT COUNT(*) FROM clip WHERE marked = 1")
        marked_count = cursor.fetchone()[0]
        print(f"\nMarked clips (selected=true): {marked_count}")
        
        return True
        
    except Exception as e:
        print(f"Database error: {e}")
        return False
    
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """Main function to handle command line arguments."""
    # Default paths
    json_file = "scripts/clips_metadata.json"
    db_path = "audio/annotations.db"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    if len(sys.argv) > 2:
        db_path = sys.argv[2]
    
    print("=" * 60)
    print("Filling Audio Clips Database")
    print("=" * 60)
    
    # Show example of path conversion
    example_path = "./audios\\routine_63_clip_012481.mp3"
    converted_path = clean_audio_path(example_path)
    print("Path conversion example:")
    print(f"  Original: {example_path}")
    print(f"  Converted: {converted_path}")
    print()
    
    # Confirm operation
    response = input(f"This will CLEAR the existing database and fill it with clips from {json_file}.\nContinue? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Fill database
    success = fill_database_from_json(json_file, db_path)
    
    if success:
        print("\n✅ Database filled successfully!")
        print(f"You can now run: python print_db.py {db_path}")
    else:
        print("\n❌ Failed to fill database.")
        sys.exit(1)

if __name__ == "__main__":
    main()
