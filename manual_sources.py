import os
import re
from datetime import datetime

print("=== MANUAL SOURCE MARKING FOR PMI DOCUMENTS ===")

# First, let's examine your file structure
with open('data/pmp_combined.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"File has {len(lines):,} lines")

# Look for natural book boundaries
print("\nLooking for book boundaries...")
boundaries = []
for i, line in enumerate(lines[:200]):  # Check first 200 lines
    line_clean = line.strip()
    if line_clean and any(indicator in line_clean.lower() for indicator in [
        'pmbok', 'practice standard', 'guide to', 'standard for', 'chapter 1', 'introduction'
    ]):
        boundaries.append((i+1, line_clean[:100]))

print("Potential book boundaries found:")
for line_num, content in boundaries[:15]:
    print(f"  Line {line_num:3d}: {content}")

# Create a semi-automatic marked version
print("\nCreating semi-automatic marked version...")

enhanced_lines = []
current_book = "PMBOK Guide 7th Edition"
enhanced_lines.append(f"[BOOK_START: {current_book}]")

for i, line in enumerate(lines):
    line_clean = line.strip().lower()
    
    # Detect book changes (you can refine these patterns)
    new_book = None
    
    if i < 50:  # Skip detection in first 50 lines to avoid false positives
        enhanced_lines.append(line.rstrip())
        continue
    
    # Look for clear book indicators
    if 'practice standard' in line_clean and 'configuration' in line_clean:
        new_book = "Practice Standard for Project Configuration Management"
    elif 'earned value management' in line_clean and len(line_clean) < 100:
        new_book = "Standard for Earned Value Management"
    elif 'work breakdown structures' in line_clean and len(line_clean) < 100:
        new_book = "Practice Standard Work Breakdown Structures"
    elif 'risk management' in line_clean and 'standard' in line_clean and len(line_clean) < 100:
        new_book = "Standard for Risk Management"
    elif 'business analysis' in line_clean and 'pmi' in line_clean and len(line_clean) < 100:
        new_book = "PMI Guide to Business Analysis"
    elif 'scheduling' in line_clean and 'practice' in line_clean and len(line_clean) < 100:
        new_book = "Practice Standard for Scheduling"
    elif 'estimating' in line_clean and 'practice' in line_clean and len(line_clean) < 100:
        new_book = "Practice Standard for Project Estimating"
    # Add more patterns as needed
    
    if new_book and new_book != current_book:
        enhanced_lines.append("")
        enhanced_lines.append(f"[BOOK_START: {new_book}]")
        enhanced_lines.append("")
        current_book = new_book
        print(f"  Added marker for: {new_book}")
    
    enhanced_lines.append(line.rstrip())

# Save the marked version
backup_name = f"data/pmp_combined_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
os.rename('data/pmp_combined.txt', backup_name)

with open('data/pmp_enhanced_marked.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(enhanced_lines))

print(f"\nFiles created:")
print(f"  Original backed up to: {backup_name}")
print(f"  Enhanced version: data/pmp_enhanced_marked.txt")

print(f"\nNext steps:")
print(f"1. Review data/pmp_enhanced_marked.txt")
print(f"2. Manually add more [BOOK_START: Book Name] markers where needed")
print(f"3. Run the vector rebuild script")
