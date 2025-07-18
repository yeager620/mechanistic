#!/usr/bin/env python3
"""
Verify the contents of the extracted H5 file
"""
import h5py
import numpy as np

def verify_h5_contents():
    """Verify the H5 file contents"""
    with h5py.File('phi2_weights.h5', 'r') as f:
        print("=== H5 FILE METADATA ===")
        print(f"File attributes: {dict(f.attrs)}")
        
        print("\n=== ALL KEYS IN H5 FILE ===")
        all_keys = list(f.keys())
        print(f"Total keys: {len(all_keys)}")
        
        # Group by prefix
        ff1_keys = [k for k in all_keys if k.startswith('FF1_') and not k.startswith('FF1_bias')]
        ff2_keys = [k for k in all_keys if k.startswith('FF2_') and not k.startswith('FF2_bias')]
        ff1_bias_keys = [k for k in all_keys if k.startswith('FF1_bias_')]
        ff2_bias_keys = [k for k in all_keys if k.startswith('FF2_bias_')]
        
        print(f"FF1 weight keys: {len(ff1_keys)}")
        print(f"FF2 weight keys: {len(ff2_keys)}")
        print(f"FF1 bias keys: {len(ff1_bias_keys)}")
        print(f"FF2 bias keys: {len(ff2_bias_keys)}")
        
        print(f"\nFF1 indices: {sorted([int(k.split('_')[1]) for k in ff1_keys])}")
        print(f"FF2 indices: {sorted([int(k.split('_')[1]) for k in ff2_keys])}")
        print(f"FF1 bias indices: {sorted([int(k.split('_')[2]) for k in ff1_bias_keys])}")
        print(f"FF2 bias indices: {sorted([int(k.split('_')[2]) for k in ff2_bias_keys])}")
        
        print(f"\nMax FF1 index: {max([int(k.split('_')[1]) for k in ff1_keys])}")
        print(f"Max FF2 index: {max([int(k.split('_')[1]) for k in ff2_keys])}")
        
        # Check dimensions
        print(f"\n=== WEIGHT DIMENSIONS ===")
        if ff1_keys:
            ff1_shape = f[ff1_keys[0]].shape
            print(f"FF1 shape: {ff1_shape}")
        if ff2_keys:
            ff2_shape = f[ff2_keys[0]].shape
            print(f"FF2 shape: {ff2_shape}")
        if ff1_bias_keys:
            ff1_bias_shape = f[ff1_bias_keys[0]].shape
            print(f"FF1 bias shape: {ff1_bias_shape}")
        if ff2_bias_keys:
            ff2_bias_shape = f[ff2_bias_keys[0]].shape
            print(f"FF2 bias shape: {ff2_bias_shape}")
        
        # Check other keys
        other_keys = [k for k in all_keys if not k.startswith(('FF1_', 'FF2_'))]
        print(f"\nOther keys: {len(other_keys)}")
        for key in other_keys[:10]:  # Show first 10
            print(f"  {key}: {f[key].shape}")

if __name__ == "__main__":
    verify_h5_contents()