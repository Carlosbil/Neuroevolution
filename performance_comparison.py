#!/usr/bin/env python3
"""
Performance comparison between sequential and threaded evaluation.
This demonstrates the performance benefits of the threading implementation.
"""

import threading
import time
import copy
import random
from datetime import datetime

def mock_evaluate_fitness(genome_id, processing_time=1.0):
    """Mock fitness evaluation that simulates training time."""
    print(f"    Training {genome_id}...")
    time.sleep(processing_time)
    fitness = random.uniform(70.0, 95.0)
    print(f"    {genome_id} achieved {fitness:.2f}% fitness")
    return fitness

def sequential_evaluation(population, processing_time=1.0):
    """Sequential evaluation (original implementation)."""
    print("=== SEQUENTIAL EVALUATION ===")
    start_time = time.time()
    
    results = []
    for i, genome in enumerate(population):
        print(f"Evaluating {genome['id']} ({i+1}/{len(population)})")
        fitness = mock_evaluate_fitness(genome['id'], processing_time)
        genome_copy = copy.deepcopy(genome)
        genome_copy['fitness'] = fitness
        results.append(genome_copy)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Sequential evaluation completed in {total_time:.2f} seconds")
    return results, total_time

def threaded_evaluation(population, processing_time=1.0, num_threads=2):
    """Threaded evaluation (new implementation)."""
    print(f"=== THREADED EVALUATION ({num_threads} THREADS) ===")
    start_time = time.time()
    
    # Shared data structures
    pending_genomes = population.copy()
    completed_genomes = []
    evaluation_log = []
    
    # Thread synchronization
    genome_lock = threading.Lock()
    results_lock = threading.Lock()
    
    def thread_worker(thread_name):
        operations_count = 0
        
        while True:
            current_genome = None
            
            # Extract genome safely
            with genome_lock:
                if pending_genomes:
                    current_genome = pending_genomes.pop(0)
                    operations_count += 1
                else:
                    break
            
            # Process genome
            if current_genome:
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                
                print(f"[{timestamp}] {thread_name} - Processing {current_genome['id']}")
                fitness = mock_evaluate_fitness(current_genome['id'], processing_time)
                
                # Store results
                genome_copy = copy.deepcopy(current_genome)
                genome_copy['fitness'] = fitness
                
                with genome_lock:
                    completed_genomes.append(genome_copy)
                
                # Log operation
                with results_lock:
                    evaluation_log.append({
                        'thread': thread_name,
                        'genome': current_genome['id'],
                        'fitness': fitness,
                        'timestamp': timestamp
                    })
        
        print(f"{thread_name} processed {operations_count} genomes")
    
    # Create and start threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=thread_worker, args=(f"THREAD-{i+1}",))
        thread.daemon = True
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Show thread distribution
    print(f"\nThread distribution:")
    for i in range(num_threads):
        thread_name = f"THREAD-{i+1}"
        thread_ops = [op for op in evaluation_log if op['thread'] == thread_name]
        print(f"  {thread_name}: {len(thread_ops)} genomes")
    
    print(f"Threaded evaluation completed in {total_time:.2f} seconds")
    return completed_genomes, total_time

def performance_comparison():
    """Compare sequential vs threaded performance."""
    print("PERFORMANCE COMPARISON: Sequential vs Threaded Evaluation")
    print("=" * 70)
    
    # Create test population
    population_size = 6
    processing_time = 0.8  # Seconds per genome
    
    test_population = []
    for i in range(population_size):
        genome = {
            'id': f'genome_{i+1}',
            'num_conv_layers': random.randint(1, 3),
            'num_fc_layers': random.randint(1, 2),
            'fitness': 0.0
        }
        test_population.append(genome)
    
    print(f"Test population: {population_size} genomes")
    print(f"Processing time per genome: {processing_time} seconds")
    print(f"Expected sequential time: ~{population_size * processing_time:.1f} seconds")
    print(f"Expected threaded time (2 threads): ~{(population_size * processing_time) / 2:.1f} seconds")
    print()
    
    # Test sequential evaluation
    population_copy1 = copy.deepcopy(test_population)
    seq_results, seq_time = sequential_evaluation(population_copy1, processing_time)
    
    print()
    
    # Test threaded evaluation
    population_copy2 = copy.deepcopy(test_population)
    thread_results, thread_time = threaded_evaluation(population_copy2, processing_time, 2)
    
    print()
    
    # Calculate improvement
    speedup = seq_time / thread_time if thread_time > 0 else 0
    time_saved = seq_time - thread_time
    efficiency = (speedup / 2) * 100  # Efficiency as percentage (2 = number of threads)
    
    print("=== PERFORMANCE RESULTS ===")
    print(f"Sequential time:     {seq_time:.2f} seconds")
    print(f"Threaded time:       {thread_time:.2f} seconds")
    print(f"Time saved:          {time_saved:.2f} seconds")
    print(f"Speedup:             {speedup:.2f}x")
    print(f"Threading efficiency: {efficiency:.1f}%")
    
    # Verify results integrity
    seq_ids = sorted([g['id'] for g in seq_results])
    thread_ids = sorted([g['id'] for g in thread_results])
    
    print(f"\n=== INTEGRITY CHECK ===")
    print(f"Sequential results:  {len(seq_results)} genomes")
    print(f"Threaded results:    {len(thread_results)} genomes")
    print(f"All genomes processed: {seq_ids == thread_ids}")
    print(f"No duplicates:       {len(set(thread_ids)) == len(thread_ids)}")
    
    # Expected performance
    theoretical_speedup = min(2.0, population_size / 2.0)  # Max 2x with 2 threads
    performance_ok = speedup >= (theoretical_speedup * 0.8)  # Allow 20% overhead
    
    print(f"\n=== PERFORMANCE ANALYSIS ===")
    print(f"Theoretical max speedup: {theoretical_speedup:.2f}x")
    print(f"Actual speedup:          {speedup:.2f}x")
    print(f"Performance acceptable:  {performance_ok}")
    
    if speedup < 1.5:
        print("⚠️  Note: Speedup lower than expected. This could be due to:")
        print("   - Thread overhead")
        print("   - System load")
        print("   - Small test population")
    else:
        print("✅ Threading implementation shows good performance improvement!")
    
    return speedup >= 1.3  # Require at least 30% improvement

if __name__ == "__main__":
    success = performance_comparison()
    print(f"\nPerformance test: {'PASSED' if success else 'FAILED'}")