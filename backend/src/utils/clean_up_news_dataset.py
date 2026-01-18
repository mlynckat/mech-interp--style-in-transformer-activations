"""
Clean up news dataset by removing web scraping artifacts.

This script:
1. Filters to only include specified authors
2. Removes duplicate title pattern at the beginning
3. Removes HuffPost editor's notes (boilerplate disclaimers)
4. Cleans Twitter/social media URL artifacts
5. Removes trailing update/correction notes
6. Removes membership/subscription calls
7. Removes duplicate articles (based on text similarity)
8. Filters out very short texts (< 100 words)
9. Recalculates word count after cleaning
10. Logs all removed parts for verification
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# Target authors to keep
TARGET_AUTHORS = [
    'Amanda Terkel',
    'Sam Levine', 
    'Paige Lavender',
    'Lee Moran',
    'Igor Bobic',
    'Mary Papenfuss',
    'Daniel Marans',
    'Marina Fang'
]

# Minimum word count threshold
MIN_WORD_COUNT = 100


@dataclass
class RemovalLog:
    """Tracks all removed content for a single article."""
    article_index: int
    author: str
    duplicate_title: Optional[str] = None
    editors_notes: List[str] = field(default_factory=list)
    twitter_urls: List[str] = field(default_factory=list)
    trailing_artifacts: List[str] = field(default_factory=list)
    
    def has_removals(self) -> bool:
        return (
            self.duplicate_title is not None or
            len(self.editors_notes) > 0 or
            len(self.twitter_urls) > 0 or
            len(self.trailing_artifacts) > 0
        )
    
    def to_dict(self) -> dict:
        return {
            'article_index': self.article_index,
            'author': self.author,
            'duplicate_title': self.duplicate_title,
            'editors_notes': self.editors_notes,
            'twitter_urls': self.twitter_urls,
            'trailing_artifacts': self.trailing_artifacts
        }


def remove_duplicate_title(text: str, log: RemovalLog) -> str:
    """
    Remove duplicate title pattern at the start of text.
    Pattern: "Title. Title. Rest of text" -> "Title. Rest of text"
    """
    # Match pattern where the same sentence appears twice at the start
    # Look for: "Sentence. Sentence. " where both sentences are identical
    match = re.match(r'^(.+?[.!?])\s*\1\s*', text)
    if match:
        log.duplicate_title = match.group(1)
        # Keep only one occurrence of the title
        return match.group(1) + ' ' + text[match.end():].lstrip()
    return text


def remove_editors_notes(text: str, log: RemovalLog) -> str:
    """
    Remove HuffPost editor's notes (boilerplate disclaimers about Trump, etc.).
    These typically start with "Editor's note:" and contain standard disclaimers.
    """
    # Pattern for HuffPost's Trump disclaimer and similar editor's notes
    # Using .*? with DOTALL flag to match across periods (like in "1.6 billion")
    patterns_with_dotall = [
        # Trump disclaimer pattern - full version ending with "entering the U.S."
        r"Editor's note: *",
    ]
    
    patterns_normal = [
        # Generic editor's note pattern (captures notes that are 1-3 sentences)
        r"Editor's [Nn]ote:(?:[^.]*\.){1,3}\s*",
        # EDITOR'S NOTE in caps
        r"EDITOR'S NOTE:(?:[^.]*\.){1,3}\s*",
    ]
    
    # Apply DOTALL patterns first (they span across periods)
    for pattern in patterns_with_dotall:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        for match in matches:
            log.editors_notes.append(match.strip())
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Apply normal patterns
    for pattern in patterns_normal:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for match in matches:
            log.editors_notes.append(match.strip())
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text


def clean_twitter_artifacts(text: str, log: RemovalLog) -> str:
    """
    Clean Twitter/social media URL artifacts while preserving context.
    Removes patterns like "pic.twitter.com/xxxxx" and "https://t.co/xxxxx"
    """
    # Find and log pic.twitter.com URLs
    pic_urls = re.findall(r'pic\.twitter\.com/\w+', text)
    log.twitter_urls.extend(pic_urls)
    
    # Find and log t.co URLs
    tco_urls = re.findall(r'https?://t\.co/\w+', text)
    log.twitter_urls.extend(tco_urls)
    
    # Remove pic.twitter.com URLs
    text = re.sub(r'\s*pic\.twitter\.com/\w+', '', text)
    
    # Remove t.co shortened URLs
    text = re.sub(r'\s*https?://t\.co/\w+', '', text)
    
    # Clean up any resulting double spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text


def remove_trailing_artifacts(text: str, log: RemovalLog) -> str:
    """
    Remove trailing update notes, corrections, and membership calls.
    """
    # Patterns to remove from end of text
    # Using .*$ to capture everything from the marker to the end (no need to find sentence end)
    trailing_patterns = [
        # Update/correction notes
        r'\s*This (?:article|story|piece|post) (?:has been|was) updated?.*$',
        r'\s*The story has been updated.*$',
        r'\s*This piece has been updated.*$',
        r'\s*A previous version.*$',
        r'\s*CORRECTION:?.*$',
        r'\s*Clarification:.*$',
        # Developing story notes
        r'\s*This is a developing story.*$',
        r'\s*Check back for updates.*$',
        r'\s*Please check back for updates.*$',
        # Membership calls and email signup
        r'\s*Your membership fuels.*$',
        r'\s*Support HuffPost.*$',
        r'\s*Support journalism.*$',
        r'\s*Enter your email address.*$',
        # Email tips solicitation
        r'\s*Email any tips.*$',
        # HuffPost campaign news solicitation
        r'\s*The Huffington Post wants to know about.*$',
        # Impact question
        r"\s*How will Trump's first 100 days impact you.*$",
        # Previously reported note
        r'\s*\w+ previously reported for.*$',
        # "contributed reporting" - flexible pattern that matches after punctuation
        r'(?<=[.!?"\'\u201d])\s*(?:\S+\s+){1,6}contributed\s+reporting.*$',
        # HuffPost app promotion
        r'\s*For more from The Huffington Post, download our app.*$',
    ]
    
    # Apply patterns iteratively (some texts have multiple trailing artifacts)
    changed = True
    while changed:
        changed = False
        for pattern in trailing_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                log.trailing_artifacts.append(match.group(0).strip())
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                changed = True
    
    return text.strip()


def clean_text(text: str, log: RemovalLog) -> str:
    """
    Apply all cleaning steps to a text.
    """
    text = remove_duplicate_title(text, log)
    text = remove_editors_notes(text, log)
    text = clean_twitter_artifacts(text, log)
    text = remove_trailing_artifacts(text, log)
    
    # Final cleanup: normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def find_duplicates(data: List[dict]) -> tuple[List[dict], List[dict]]:
    """
    Find and remove duplicate articles based on text content.
    Returns (unique_articles, duplicate_articles).
    
    Uses first 500 characters of text as a fingerprint to detect duplicates.
    """
    seen_fingerprints = {}
    unique = []
    duplicates = []
    
    for item in data:
        text = item.get('text', '')
        # Use first 500 chars as fingerprint (after normalizing whitespace)
        fingerprint = re.sub(r'\s+', ' ', text[:500]).strip().lower()
        
        if fingerprint in seen_fingerprints:
            # This is a duplicate
            duplicates.append({
                'duplicate_of_index': seen_fingerprints[fingerprint],
                'author': item.get('style'),
                'text_preview': text[:200] + '...'
            })
        else:
            seen_fingerprints[fingerprint] = len(unique)
            unique.append(item)
    
    return unique, duplicates


def clean_dataset(input_path: str, output_path: str, log_path: str) -> dict:
    """
    Clean the dataset and save to a new file.
    
    Returns statistics about the cleaning process.
    """
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total entries in original dataset: {len(data)}")
    
    # Filter by target authors
    filtered_by_author = [item for item in data if item.get('style') in TARGET_AUTHORS]
    print(f"Entries from target authors: {len(filtered_by_author)}")
    
    # Clean texts and update word counts
    cleaned_data = []
    removed_short = 0
    removal_logs = []
    
    for idx, item in enumerate(filtered_by_author):
        original_text = item.get('text', '')
        
        # Create log for this article
        log = RemovalLog(
            article_index=idx,
            author=item.get('style', 'Unknown')
        )
        
        cleaned_text = clean_text(original_text, log)
        new_word_count = count_words(cleaned_text)
        
        # Save log if anything was removed
        if log.has_removals():
            removal_logs.append(log.to_dict())
        
        # Filter out very short texts
        if new_word_count < MIN_WORD_COUNT:
            removed_short += 1
            continue
        
        cleaned_item = {
            'style': item.get('style'),
            'text': cleaned_text,
            'category': item.get('category'),
            'text_length_words': new_word_count
        }
        cleaned_data.append(cleaned_item)
    
    print(f"Removed {removed_short} texts shorter than {MIN_WORD_COUNT} words")
    print(f"After length filter: {len(cleaned_data)}")
    
    # Find and remove duplicates
    print("\nChecking for duplicate articles...")
    unique_data, duplicate_articles = find_duplicates(cleaned_data)
    print(f"Found {len(duplicate_articles)} duplicate articles")
    print(f"After duplicate removal: {len(unique_data)}")
    
    # Count by author
    print("\nArticles per author:")
    for author in TARGET_AUTHORS:
        count = sum(1 for item in unique_data if item.get('style') == author)
        print(f"  {author}: {count}")
    
    # Save cleaned dataset
    print(f"\nSaving cleaned dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)
    
    # Save removal log
    print(f"Saving removal log to {log_path}...")
    log_data = {
        'summary': {
            'total_articles_with_removals': len(removal_logs),
            'total_duplicate_titles_removed': sum(1 for log in removal_logs if log['duplicate_title']),
            'total_editors_notes_removed': sum(len(log['editors_notes']) for log in removal_logs),
            'total_twitter_urls_removed': sum(len(log['twitter_urls']) for log in removal_logs),
            'total_trailing_artifacts_removed': sum(len(log['trailing_artifacts']) for log in removal_logs),
            'total_duplicate_articles_removed': len(duplicate_articles)
        },
        'content_removals': removal_logs,
        'duplicate_articles': duplicate_articles
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    stats = {
        'original_count': len(data),
        'target_authors_count': len(filtered_by_author),
        'removed_short': removed_short,
        'removed_duplicates': len(duplicate_articles),
        'final_count': len(unique_data),
        'articles_with_removals': len(removal_logs)
    }
    
    return stats


def main():
    # Define paths
    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'news_authormix_datasets'
    input_path = data_dir / 'POLITICS_authormix.json'
    output_path = data_dir / 'POLITICS_authormix_cleaned.json'
    log_path = data_dir / 'POLITICS_authormix_cleanup_log.json'
    
    print("=" * 60)
    print("News Dataset Cleanup")
    print("=" * 60)
    print(f"\nTarget authors: {', '.join(TARGET_AUTHORS)}")
    print(f"Minimum word count: {MIN_WORD_COUNT}")
    print()
    
    stats = clean_dataset(str(input_path), str(output_path), str(log_path))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Original dataset: {stats['original_count']} entries")
    print(f"After author filter: {stats['target_authors_count']} entries")
    print(f"Removed (too short): {stats['removed_short']} entries")
    print(f"Removed (duplicates): {stats['removed_duplicates']} entries")
    print(f"Final cleaned dataset: {stats['final_count']} entries")
    print(f"\nArticles with content removals: {stats['articles_with_removals']}")
    print(f"\nOutput saved to: {output_path}")
    print(f"Removal log saved to: {log_path}")


if __name__ == '__main__':
    main()
