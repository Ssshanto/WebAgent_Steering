#!/usr/bin/env python3
"""
Ground truth extraction heuristics for expanded action space tasks.

Provides oracle functions for:
- click-checkboxes (multi-selection)
- click-option (radio buttons)
- choose-list (dropdowns)
- choose-date (date pickers)
- enter-date (semantic date typing)
- enter-time (semantic time typing)
- guess-number (feedback loop/logic)
"""

import re
from typing import Dict, List, Optional, Union


def extract_click_checkboxes_truth(utterance: str, dom_elements: List[Dict]) -> List[str]:
    """
    Extract which checkboxes should be clicked.
    Example: "Select Brittany and Mia" -> find refs with text "Brittany" and "Mia"
    """
    # Common patterns: "Select X and Y", "Click X, Y and Z"
    match = re.search(r'Select (.+)', utterance, re.IGNORECASE)
    if not match:
        return []
    
    items_str = match.group(1)
    # Split on "and", ","
    items = re.split(r',\s*|\s+and\s+', items_str)
    items = [item.strip() for item in items]
    
    # Find refs for each item
    refs = []
    for item in items:
        for elem in dom_elements:
            text = (elem.get("text") or "").strip()
            if item.lower() in text.lower():
                refs.append(f"click ref={elem['ref']}")
                break
    
    return refs


def extract_click_option_truth(utterance: str, dom_elements: List[Dict]) -> Optional[str]:
    """
    Extract which radio button should be selected.
    Example: "Select the blue color" -> find ref with "blue"
    """
    match = re.search(r'(?:Select|Choose) (?:the )?(.+)', utterance, re.IGNORECASE)
    if not match:
        return None
    
    target = match.group(1).strip()
    
    for elem in dom_elements:
        text = (elem.get("text") or "").strip()
        if target.lower() in text.lower():
            return f"click ref={elem['ref']}"
    
    return None


def extract_choose_list_truth(utterance: str, dom_elements: List[Dict]) -> Optional[str]:
    """
    Extract dropdown selection.
    Example: "Select New York from the dropdown" -> select ref=X option="New York"
    """
    match = re.search(r'Select (.+?) from', utterance, re.IGNORECASE)
    if not match:
        return None
    
    option = match.group(1).strip()
    
    # Find dropdown element (select tag)
    for elem in dom_elements:
        tag = elem.get("tag", "")
        if tag == "select":
            return f'select ref={elem["ref"]} option="{option}"'
    
    return None


def extract_choose_date_truth(utterance: str, dom_elements: List[Dict]) -> Optional[str]:
    """
    Extract date selection from date picker.
    Example: "Select 03/15/2024"
    """
    match = re.search(r'Select (\d{2}/\d{2}/\d{4})', utterance)
    if not match:
        return None
    
    date = match.group(1)
    
    for elem in dom_elements:
        tag = elem.get("tag", "")
        if tag == "input" and "date" in elem.get("classes", "").lower():
            return f'type ref={elem["ref"]} text="{date}"'
    
    return None


def extract_enter_date_truth(utterance: str, dom_elements: List[Dict]) -> Optional[str]:
    """
    Extract date to type into field.
    Example: "Enter the date 12/25/2023"
    """
    match = re.search(r'Enter (?:the date )?(\d{2}/\d{2}/\d{4})', utterance, re.IGNORECASE)
    if not match:
        return None
    
    date = match.group(1)
    
    for elem in dom_elements:
        tag = elem.get("tag", "")
        if tag == "input":
            return f'type ref={elem["ref"]} text="{date}"'
    
    return None


def extract_enter_time_truth(utterance: str, dom_elements: List[Dict]) -> Optional[str]:
    """
    Extract time to type into field.
    Example: "Enter the time 14:30"
    """
    match = re.search(r'Enter (?:the time )?(\d{1,2}:\d{2})', utterance, re.IGNORECASE)
    if not match:
        return None
    
    time = match.group(1)
    
    for elem in dom_elements:
        tag = elem.get("tag", "")
        if tag == "input":
            return f'type ref={elem["ref"]} text="{time}"'
    
    return None


def extract_guess_number_truth(utterance: str, dom_elements: List[Dict]) -> Optional[str]:
    """
    Extract number guessing strategy (binary search midpoint).
    Example: "Guess the number between 0-100" -> start with 50
    """
    match = re.search(r'between (\d+)-(\d+)', utterance, re.IGNORECASE)
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        guess = (low + high) // 2
        
        for elem in dom_elements:
            tag = elem.get("tag", "")
            if tag == "input":
                return f'type ref={elem["ref"]} text="{guess}"'
    
    return None


def extract_ground_truth(task_name: str, utterance: str, dom_elements: List[Dict]) -> Union[str, List[str], None]:
    """
    Main function to extract ground truth for expanded tasks.
    Returns single action string or list of action strings.
    """
    extractors = {
        "click-checkboxes": extract_click_checkboxes_truth,
        "click-option": extract_click_option_truth,
        "choose-list": extract_choose_list_truth,
        "choose-date": extract_choose_date_truth,
        "enter-date": extract_enter_date_truth,
        "enter-time": extract_enter_time_truth,
        "guess-number": extract_guess_number_truth,
    }
    
    extractor = extractors.get(task_name)
    if extractor:
        return extractor(utterance, dom_elements)
    
    return None


if __name__ == "__main__":
    print("="*60)
    print("Ground Truth Extraction for Expanded Action Space")
    print("="*60)
    print()
    print("Supported tasks:")
    tasks = [
        "click-checkboxes", "click-option", "choose-list",
        "choose-date", "enter-date", "enter-time", "guess-number"
    ]
    for task in tasks:
        print(f"  ✓ {task}")
    print()
    print("Usage:")
    print("  from extract_ground_truth_expanded import extract_ground_truth")
    print("  truth = extract_ground_truth(task_name, utterance, dom_elements)")
