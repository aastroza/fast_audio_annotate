#!/usr/bin/env python3
"""Simple script to read and print the clips database content."""

import sqlite3
import sys
from pathlib import Path

def print_db_content(db_path):
    """Print the content of the clips database."""
    db_file = Path(db_path)
    
    if not db_file.exists():
        print(f"Database file not found: {db_path}")
        return
    
    print(f"Reading database: {db_path}")
    print("=" * 80)
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in database: {[table[0] for table in tables]}")
        print()
        
        # Read clips table
        cursor.execute("SELECT * FROM clip ORDER BY audio_path, start_timestamp")
        rows = cursor.fetchall()
        
        # Get column names
        cursor.execute("PRAGMA table_info(clip)")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"Columns: {columns}")
        print()
        
        print(f"Total records: {len(rows)}")
        print("-" * 80)
        
        # Print header
        header = " | ".join(f"{col:15}" for col in columns)
        print(header)
        print("-" * len(header))
        
        # Print rows with better formatting for clips
        for row in rows:
            # Format specific columns for better readability
            formatted_row = []
            for i, val in enumerate(row):
                if columns[i] in ['start_timestamp', 'end_timestamp'] and val is not None:
                    # Format timestamps to 2 decimal places
                    formatted_row.append(f"{float(val):.2f}")
                elif columns[i] == 'text' and val:
                    # Truncate text and show preview
                    text_preview = str(val)[:30] + "..." if len(str(val)) > 30 else str(val)
                    formatted_row.append(text_preview)
                else:
                    formatted_row.append(str(val) if val is not None else "")
            
            row_str = " | ".join(f"{val[:15]:15}" for val in formatted_row)
            print(row_str)
            
        print("-" * 80)
        
        # Summary statistics for clips
        cursor.execute("SELECT COUNT(DISTINCT audio_path) FROM clip")
        unique_audio_files = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM clip WHERE marked = 1")
        marked_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM clip WHERE text = '' OR text IS NULL")
        empty_transcription_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT audio_path, COUNT(*) FROM clip GROUP BY audio_path ORDER BY audio_path")
        clips_per_audio = cursor.fetchall()
        
        cursor.execute("SELECT AVG(end_timestamp - start_timestamp) FROM clip")
        avg_clip_duration = cursor.fetchone()[0]
        
        print("\nSummary:")
        print(f"- Total clips: {len(rows)}")
        print(f"- Unique audio files: {unique_audio_files}")
        print(f"- Clips with empty transcription: {empty_transcription_count}")
        print(f"- Marked clips: {marked_count}")
        if avg_clip_duration:
            print(f"- Average clip duration: {avg_clip_duration:.2f} seconds")
        print("\nClips per audio file:")
        for audio_path, count in clips_per_audio:
            print(f"  {audio_path}: {count} clips")
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    # Default path
    db_path = "audio/annotations.db"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    print_db_content(db_path)