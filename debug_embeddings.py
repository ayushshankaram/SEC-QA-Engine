#!/usr/bin/env python3
"""
Debug Embedding Ensemble - Check what models are active
"""

import sys
import os
sys.path.append('/Users/ayushshankaram/Desktop/QAEngine/src')

import logging
from core.embedding_ensemble import EmbeddingEnsemble

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🔧 Testing EmbeddingEnsemble initialization...")
    
    try:
        ensemble = EmbeddingEnsemble()
        
        print(f"\n📊 Active clients: {list(ensemble.clients.keys())}")
        print(f"📊 Model weights: {ensemble.weights}")
        print(f"📊 Enabled models: {ensemble.enable_models}")
        
        # Test embedding a simple query
        test_query = "What are the revenue sources?"
        print(f"\n🧪 Testing embedding for: '{test_query}'")
        
        result = ensemble.embed_query(test_query)
        
        if result:
            print(f"✅ Embedding successful! Dimension: {len(result)}")
            print(f"🔢 First 5 values: {result[:5]}")
        else:
            print("❌ Embedding failed - returned None")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
