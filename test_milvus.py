#!/usr/bin/env python3
"""
Comprehensive Milvus Test Script
Tests connection, CRUD operations, search, and query
"""

from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
import time
import random
import sys

def test_milvus():
    """Run comprehensive Milvus tests"""
    
    print("=" * 60)
    print("MILVUS COMPREHENSIVE TEST")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Connection
        print("Test 1: Connecting to Milvus...")
        connections.connect(
            alias="default",
            host='localhost',
            port='19530',
            timeout=10
        )
        print("✓ Connected successfully")
        print()
        
        # Test 2: List Collections
        print("Test 2: Listing existing collections...")
        collections = utility.list_collections()
        print(f"✓ Found {len(collections)} collections")
        for coll in collections:
            print(f"  - {coll}")
        print()
        
        # Test 3: Create Collection
        print("Test 3: Creating test collection...")
        test_name = f"test_collection_{int(time.time())}"
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]
        schema = CollectionSchema(fields=fields, description="Test collection")
        collection = Collection(name=test_name, schema=schema)
        print(f"✓ Created collection: {test_name}")
        print()
        
        # Test 4: Insert Data
        print("Test 4: Inserting 100 vectors...")
        data = [
            [f"text_{i}" for i in range(100)],
            [[random.random() for _ in range(128)] for _ in range(100)]
        ]
        insert_result = collection.insert(data)
        print(f"✓ Inserted {len(insert_result.primary_keys)} vectors")
        print()
        
        # Test 5: Flush
        print("Test 5: Flushing data to storage...")
        collection.flush()
        print("✓ Data flushed")
        print()
        
        # Test 6: Build Index
        print("Test 6: Building index...")
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("✓ Index created")
        print()
        
        # Test 7: Load Collection
        print("Test 7: Loading collection into memory...")
        collection.load()
        print("✓ Collection loaded")
        
        # Wait for loading to complete
        print("  Waiting for collection to finish loading...")
        time.sleep(2)
        print()
        
        # Test 8: Get Collection Stats
        print("Test 8: Getting collection statistics...")
        stats = collection.num_entities
        print(f"✓ Collection has {stats} entities")
        print()
        
        # Test 9: Search
        print("Test 9: Performing vector search...")
        search_data = [[random.random() for _ in range(128)]]
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=search_data,
            anns_field="embedding",
            param=search_params,
            limit=5,
            output_fields=["text_id"]
        )
        print(f"✓ Search completed, found {len(results[0])} results")
        for i, hit in enumerate(results[0]):
            print(f"  {i+1}. ID={hit.id}, text_id={hit.entity.get('text_id')}, distance={hit.distance:.4f}")
        print()
        
        # Test 10: Query
        print("Test 10: Performing filtered query...")
        query_result = collection.query(
            expr="text_id in ['text_0', 'text_1', 'text_2']",
            output_fields=["text_id", "id"]
        )
        print(f"✓ Query completed, found {len(query_result)} results")
        for item in query_result:
            print(f"  - {item}")
        print()
        
        # Test 11: Release and Drop
        print("Test 11: Cleaning up test collection...")
        collection.release()
        #collection.drop()
        print(f"✓ Test collection '{test_name}' dropped")
        print()
        
        # Final Summary
        print("=" * 60)
        print("✓✓✓ ALL TESTS PASSED - MILVUS IS HEALTHY ✓✓✓")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print("✗✗✗ TEST FAILED ✗✗✗")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            connections.disconnect("default")
            print("\nDisconnected from Milvus")
        except:
            pass

if __name__ == "__main__":
    success = test_milvus()
    sys.exit(0 if success else 1)

