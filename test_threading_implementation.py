#!/usr/bin/env python3
"""
Test script to verify that the threading implementation works correctly.
This script tests the key threading functionality without running the full neuroevolution.
"""

import threading
import time
import copy
import random
from datetime import datetime
import numpy as np

def test_genome_extraction_threading():
    """Test the threading pattern used in the hybrid neuroevolution notebook."""
    
    print("Testing genome extraction threading pattern...")
    print("=" * 60)
    
    # Simulate a population of genomes
    test_population = []
    for i in range(6):  # Small test population
        genome = {
            'id': f'test_genome_{i+1}',
            'num_conv_layers': random.randint(1, 3),
            'num_fc_layers': random.randint(1, 2),
            'optimizer': random.choice(['adam', 'sgd']),
            'learning_rate': random.choice([0.001, 0.01]),
            'fitness': 0.0
        }
        test_population.append(genome)
    
    print(f"Initial test population: {len(test_population)} genomes")
    for genome in test_population:
        print(f"   {genome['id']}: {genome['num_conv_layers']} conv, {genome['num_fc_layers']} fc")
    
    # Threading variables (similar to the notebook implementation)
    pending_genomes = test_population.copy()
    completed_genomes = []
    fitness_scores = []
    best_fitness_so_far = 0.0
    
    # Thread synchronization locks
    genome_lock = threading.Lock()
    results_lock = threading.Lock()
    evaluation_log = []
    
    def mock_evaluate_fitness(genome):
        """Mock fitness evaluation - simulates training time."""
        print(f"         [MOCK] Training {genome['id']}...")
        time.sleep(random.uniform(0.5, 1.5))  # Simulate training time
        fitness = random.uniform(70.0, 95.0)  # Mock fitness score
        print(f"         [MOCK] {genome['id']} achieved {fitness:.2f}% fitness")
        return fitness
    
    def thread_worker(thread_name):
        """Thread worker function that processes genomes from the shared queue."""
        operations_count = 0
        
        print(f"      🚀 {thread_name} started")
        
        while True:
            # Extract genome from pending list safely
            current_genome = None
            remaining_count = 0
            completed_count = 0
            
            # Critical section: extract genome from pending list
            with genome_lock:
                if pending_genomes:  # If there are genomes to process
                    current_genome = pending_genomes.pop(0)  # Extract first genome
                    remaining_count = len(pending_genomes)
                    completed_count = len(completed_genomes)
                    operations_count += 1
                else:
                    # No more genomes to process
                    break
            
            # Process the genome outside the lock
            if current_genome:
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                
                # Log extraction
                with results_lock:
                    evaluation_log.append({
                        'thread': thread_name,
                        'operation': f'EXTRACTED {current_genome["id"]}',
                        'pending_remaining': remaining_count,
                        'completed_count': completed_count,
                        'timestamp': timestamp
                    })
                
                print(f"      [{timestamp}] {thread_name} - Processing genome {current_genome['id']}")
                print(f"         Remaining: {remaining_count} | Completed: {completed_count + 1}")
                
                # Evaluate fitness (this is the time-consuming part)
                fitness = mock_evaluate_fitness(current_genome)
                current_genome['fitness'] = fitness
                
                # Store results safely
                with genome_lock:
                    completed_genomes.append(current_genome)
                    fitness_scores.append(fitness)
                    
                    # Update best fitness if necessary
                    nonlocal best_fitness_so_far
                    if fitness > best_fitness_so_far:
                        best_fitness_so_far = fitness
                        print(f"         🎯 New best fitness: {fitness:.2f}%!")
                
                # Log completion
                completion_timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                with results_lock:
                    evaluation_log.append({
                        'thread': thread_name,
                        'operation': f'COMPLETED {current_genome["id"]} - {fitness:.2f}%',
                        'pending_remaining': len(pending_genomes),
                        'completed_count': len(completed_genomes),
                        'timestamp': completion_timestamp
                    })
                
                print(f"         ✅ Fitness: {fitness:.2f}% | Best so far: {best_fitness_so_far:.2f}%")
        
        print(f"      🏁 {thread_name} finished - {operations_count} genomes processed")
    
    # Test the threading implementation
    print(f"\nStarting threaded evaluation with {len(pending_genomes)} genomes...")
    
    # Create and start 2 worker threads
    thread1 = threading.Thread(target=thread_worker, args=("THREAD-1",))
    thread2 = threading.Thread(target=thread_worker, args=("THREAD-2",))
    
    # Mark threads as daemon to ensure clean shutdown
    thread1.daemon = True
    thread2.daemon = True
    
    # Start both threads
    start_time = time.time()
    thread1.start()
    thread2.start()
    
    print(f"   ✅ Threads started. Active threads: {threading.active_count()}")
    
    # Wait for both threads to complete
    thread1.join()
    thread2.join()
    
    end_time = time.time()
    
    print(f"\n💯 Threaded evaluation completed!")
    print(f"   Total genomes processed: {len(completed_genomes)}")
    print(f"   Best fitness found: {best_fitness_so_far:.2f}%")
    print(f"   Active threads: {threading.active_count()}")
    print(f"   Total time: {end_time - start_time:.2f} seconds")
    
    # Show chronological log of operations
    print(f"\n--- Chronological Thread Operations Log ---")
    evaluation_log.sort(key=lambda x: x['timestamp'])
    for entry in evaluation_log:
        print(f"[{entry['timestamp']}] {entry['thread']} - {entry['operation']}")
    
    # Verify results integrity
    print(f"\n--- Results Verification ---")
    print(f"Original population size: {len(test_population)}")
    print(f"Completed genomes: {len(completed_genomes)}")
    print(f"Fitness scores: {len(fitness_scores)}")
    print(f"All genomes processed: {len(test_population) == len(completed_genomes)}")
    print(f"No genomes lost: {len(pending_genomes) == 0}")
    
    # Check for duplicates
    completed_ids = [g['id'] for g in completed_genomes]
    original_ids = [g['id'] for g in test_population]
    
    print(f"No duplicate processing: {len(completed_ids) == len(set(completed_ids))}")
    print(f"All original IDs found: {set(original_ids) == set(completed_ids)}")
    
    # Show fitness statistics
    if fitness_scores:
        print(f"\n--- Fitness Statistics ---")
        print(f"Average fitness: {np.mean(fitness_scores):.2f}%")
        print(f"Best fitness: {np.max(fitness_scores):.2f}%")
        print(f"Worst fitness: {np.min(fitness_scores):.2f}%")
        print(f"Standard deviation: {np.std(fitness_scores):.2f}%")
    
    # Show thread distribution
    thread1_ops = [entry for entry in evaluation_log if entry['thread'] == 'THREAD-1' and 'EXTRACTED' in entry['operation']]
    thread2_ops = [entry for entry in evaluation_log if entry['thread'] == 'THREAD-2' and 'EXTRACTED' in entry['operation']]
    
    print(f"\n--- Thread Distribution ---")
    print(f"THREAD-1 processed: {len(thread1_ops)} genomes")
    print(f"THREAD-2 processed: {len(thread2_ops)} genomes")
    print(f"Total processed: {len(thread1_ops) + len(thread2_ops)} genomes")
    
    print(f"\n🎉 Threading test completed successfully!")
    return len(completed_genomes) == len(test_population)

if __name__ == "__main__":
    test_result = test_genome_extraction_threading()
    print(f"\nTest result: {'PASSED' if test_result else 'FAILED'}")