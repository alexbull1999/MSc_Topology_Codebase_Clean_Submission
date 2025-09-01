#!/usr/bin/env python3
"""
Check SBERT Token Files for UIDs

This script inspects your existing SBERT token files to see if they contain
UIDs that can be matched to the original SNLI/MNLI data for fair comparison.
"""

import pickle
import argparse
from pathlib import Path

def inspect_sbert_file(file_path, dataset_name):
    """Inspect a single SBERT token file"""
    
    if not Path(file_path).exists():
        print(f"❌ {dataset_name} file not found: {file_path}")
        return None
    
    print(f"\n📁 Inspecting {dataset_name}: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"   Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            
            # Check each key and its properties
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"   {key}: shape {value.shape}")
                elif hasattr(value, '__len__'):
                    print(f"   {key}: {len(value)} items")
                    if len(value) > 0:
                        print(f"     Sample item type: {type(value[0])}")
                        if isinstance(value[0], (str, int, float)):
                            print(f"     Sample items: {value[:5]}...")
                        elif hasattr(value[0], 'keys'):
                            print(f"     Sample item keys: {list(value[0].keys())}")
                else:
                    print(f"   {key}: {type(value)}")
            
            # Look specifically for UID-related fields
            uid_fields = ['uid', 'uids', 'pairID', 'pair_id', 'sample_id', 'ids']
            found_uid_fields = []
            
            for field in uid_fields:
                if field in data:
                    found_uid_fields.append(field)
                    print(f"   🎯 Found UID field: {field}")
                    
                    # Show sample UIDs
                    uid_data = data[field]
                    if hasattr(uid_data, '__len__') and len(uid_data) > 0:
                        if hasattr(uid_data, 'shape'):
                            sample_uids = uid_data[:5]
                        else:
                            sample_uids = uid_data[:5]
                        print(f"      Sample UIDs: {sample_uids}")
            
            if not found_uid_fields:
                print(f"   ❌ No UID fields found")
                print(f"   🔍 Available fields: {list(data.keys())}")
                
                # Check if any field contains string data that might be UIDs
                for key, value in data.items():
                    if hasattr(value, '__len__') and len(value) > 0:
                        if hasattr(value, '__getitem__'):
                            try:
                                sample_item = value[0]
                                if isinstance(sample_item, str) and ('#' in sample_item or 'r1' in sample_item):
                                    print(f"   🤔 {key} might contain UIDs: {sample_item}")
                                elif isinstance(sample_item, dict) and any(uid_field in sample_item for uid_field in uid_fields):
                                    print(f"   🤔 {key} contains dicts with possible UIDs")
                                    for uid_field in uid_fields:
                                        if uid_field in sample_item:
                                            print(f"      Found {uid_field}: {sample_item[uid_field]}")
                            except:
                                pass
            
            return {
                'has_uids': len(found_uid_fields) > 0,
                'uid_fields': found_uid_fields,
                'total_samples': len(data.get('premise_tokens', data.get('tokens', []))) if 'premise_tokens' in data or 'tokens' in data else 'unknown',
                'data_structure': data.keys()
            }
            
        else:
            print(f"   ❌ Data is not a dictionary: {type(data)}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return None


def main():
    """Main inspection function"""
    
    parser = argparse.ArgumentParser(description="Check SBERT token files for UIDs")
    parser.add_argument('--snli_sbert_path', 
                       default='/vol/bitbucket/ahb24/tda_entailment_new/chunked_snli_train_sbert_tokens_chunk_1_of_5.pkl',
                       help='Path to SNLI SBERT tokens file')
    parser.add_argument('--mnli_sbert_path',
                       default='/vol/bitbucket/ahb24/tda_entailment_new/chunked_mnli_train_sbert_tokens_chunk_1_of_5.pkl',
                       help='Path to MNLI SBERT tokens file')
    
    args = parser.parse_args()
    
    print("🔍 Checking SBERT token files for UIDs...")
    print("This will determine if we can create fair UID-matched training data")
    
    # Inspect SBERT files
    snli_result = inspect_sbert_file(args.snli_sbert_path, "SNLI SBERT")
    mnli_result = inspect_sbert_file(args.mnli_sbert_path, "MNLI SBERT")


if __name__ == "__main__":
    main()