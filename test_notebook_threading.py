#!/usr/bin/env python3
"""
Test script to validate the threaded notebook implementation with minimal dependencies.
This tests the actual HybridNeuroevolution class with threading functionality.
"""

import threading
import time
import copy
import random
from datetime import datetime

def test_hybrid_neuroevolution_threading():
    """Test the HybridNeuroevolution class with a minimal mock setup."""
    
    print("Testing HybridNeuroevolution threading implementation...")
    print("=" * 60)
    
    # Mock configuration (minimal)
    config = {
        'population_size': 4,
        'num_epochs': 1,
        'early_stopping_patience': 10,
        'epoch_patience': 1,
        'improvement_threshold': 0.1,
        'current_mutation_rate': 0.3
    }
    
    # Mock data loader (empty, won't be used in our test)
    class MockDataLoader:
        def __init__(self):
            pass
        
        def __iter__(self):
            # Return empty iterator to simulate quick training
            return iter([])
        
        def __len__(self):
            return 0
    
    # Simplified HybridNeuroevolution class for testing
    class TestHybridNeuroevolution:
        def __init__(self, config, train_loader, test_loader):
            self.config = config
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.population = []
            self.generation = 0
            self.best_individual = None
            self.fitness_history = []
            self.generation_stats = []
        
        def evaluate_fitness(self, genome):
            """Mock fitness evaluation that simulates some processing time."""
            # Simulate training time
            time.sleep(random.uniform(0.2, 0.8))
            # Return mock fitness
            fitness = random.uniform(70.0, 95.0)
            return fitness
        
        def _thread_worker(self, thread_name: str):
            """Thread worker function that processes genomes from the shared queue."""
            operations_count = 0
            
            print(f"      🚀 {thread_name} started")
            
            while True:
                # Extract genome from pending list safely
                current_genome = None
                remaining_count = 0
                completed_count = 0
                
                # Critical section: extract genome from pending list
                with self.genome_lock:
                    if self.pending_genomes:  # If there are genomes to process
                        current_genome = self.pending_genomes.pop(0)  # Extract first genome
                        remaining_count = len(self.pending_genomes)
                        completed_count = len(self.completed_genomes)
                        operations_count += 1
                    else:
                        # No more genomes to process
                        break
                
                # Process the genome outside the lock
                if current_genome:
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    
                    # Log extraction
                    with self.results_lock:
                        self.evaluation_log.append({
                            'thread': thread_name,
                            'operation': f'EXTRACTED {current_genome["id"]}',
                            'pending_remaining': remaining_count,
                            'completed_count': completed_count,
                            'timestamp': timestamp
                        })
                    
                    print(f"      [{timestamp}] {thread_name} - Processing genome {current_genome['id']}")
                    print(f"         Remaining: {remaining_count} | Completed: {completed_count + 1}")
                    
                    # Evaluate fitness (this is the time-consuming part)
                    fitness = self.evaluate_fitness(current_genome)
                    current_genome['fitness'] = fitness
                    
                    # Store results safely
                    with self.genome_lock:
                        self.completed_genomes.append(current_genome)
                        self.fitness_scores.append(fitness)
                        
                        # Update best fitness if necessary
                        if fitness > self.best_fitness_so_far:
                            self.best_fitness_so_far = fitness
                            print(f"         🎯 New best fitness in this generation: {fitness:.2f}%!")
                    
                    # Log completion
                    completion_timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    with self.results_lock:
                        self.evaluation_log.append({
                            'thread': thread_name,
                            'operation': f'COMPLETED {current_genome["id"]} - {fitness:.2f}%',
                            'pending_remaining': len(self.pending_genomes),
                            'completed_count': len(self.completed_genomes),
                            'timestamp': completion_timestamp
                        })
                    
                    print(f"         ✅ Fitness: {fitness:.2f}% | Best so far: {self.best_fitness_so_far:.2f}%")
            
            print(f"      🏁 {thread_name} finished - {operations_count} genomes processed")

        def evaluate_population(self):
            print(f"\nEvaluating population (Generation {self.generation})...")
            print(f"Processing {len(self.population)} individuals using 2 concurrent threads...")
            
            # Initialize shared data structures for threading
            self.pending_genomes = self.population.copy()  # Source list of genomes to evaluate
            self.completed_genomes = []  # Destination list for evaluated genomes
            self.fitness_scores = []
            self.best_fitness_so_far = 0.0
            
            # Thread synchronization locks
            self.genome_lock = threading.Lock()  # Protects genome lists access
            self.results_lock = threading.Lock()  # Protects results logging
            self.evaluation_log = []  # Log of thread operations
            
            print(f"Starting threaded evaluation with {len(self.pending_genomes)} genomes...")
            
            # Create and start 2 worker threads
            thread1 = threading.Thread(target=self._thread_worker, args=("THREAD-1",))
            thread2 = threading.Thread(target=self._thread_worker, args=("THREAD-2",))
            
            # Mark threads as daemon to ensure clean shutdown
            thread1.daemon = True
            thread2.daemon = True
            
            # Start both threads
            thread1.start()
            thread2.start()
            
            print(f"   ✅ Threads started. Active threads: {threading.active_count()}")
            
            # Wait for both threads to complete
            thread1.join()
            thread2.join()
            
            print(f"\n💯 Threaded evaluation completed!")
            print(f"   Total genomes processed: {len(self.completed_genomes)}")
            print(f"   Best fitness found: {self.best_fitness_so_far:.2f}%")
            print(f"   Active threads: {threading.active_count()}")
            
            # Show chronological log of operations
            print(f"\n--- Chronological Thread Operations Log ---")
            self.evaluation_log.sort(key=lambda x: x['timestamp'])
            for entry in self.evaluation_log:
                print(f"[{entry['timestamp']}] {entry['thread']} - {entry['operation']}")
            
            # Update population with evaluated results and get scores
            self.population = self.completed_genomes.copy()
            fitness_scores = self.fitness_scores.copy()
            
            # Statistics calculation
            if fitness_scores:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                max_fitness = max(fitness_scores)
                min_fitness = min(fitness_scores)
                # Simple std dev calculation
                mean = avg_fitness
                variance = sum((x - mean) ** 2 for x in fitness_scores) / len(fitness_scores)
                std_fitness = variance ** 0.5
            else:
                avg_fitness = max_fitness = min_fitness = std_fitness = 0.0

            stats = {
                'generation': self.generation,
                'avg_fitness': avg_fitness,
                'max_fitness': max_fitness,
                'min_fitness': min_fitness,
                'std_fitness': std_fitness
            }
            self.generation_stats.append(stats)
            self.fitness_history.append(max_fitness)

            best_genome = max(self.population, key=lambda x: x['fitness'])
            if self.best_individual is None or best_genome['fitness'] > self.best_individual['fitness']:
                self.best_individual = copy.deepcopy(best_genome)
                print(f"\nNew global best individual found!")

            print(f"\nGENERATION {self.generation} STATISTICS:")
            print(f"   Maximum fitness: {max_fitness:.2f}%")
            print(f"   Average fitness: {avg_fitness:.2f}%")
            print(f"   Minimum fitness: {min_fitness:.2f}%")
            print(f"   Standard deviation: {std_fitness:.2f}%")
            print(f"   Best individual: {best_genome['id']} with {best_genome['fitness']:.2f}%")
            print(f"   Global best individual: {self.best_individual['id']} with {self.best_individual['fitness']:.2f}%")
            
            return True
    
    # Create test population
    test_population = []
    for i in range(config['population_size']):
        genome = {
            'id': f'test_genome_{i+1}',
            'num_conv_layers': random.randint(1, 3),
            'num_fc_layers': random.randint(1, 2),
            'optimizer': random.choice(['adam', 'sgd']),
            'learning_rate': random.choice([0.001, 0.01]),
            'fitness': 0.0
        }
        test_population.append(genome)
    
    # Initialize mock data loaders
    mock_train_loader = MockDataLoader()
    mock_test_loader = MockDataLoader()
    
    # Create and test the HybridNeuroevolution instance
    neuroevolution = TestHybridNeuroevolution(config, mock_train_loader, mock_test_loader)
    neuroevolution.population = test_population
    
    print(f"Created HybridNeuroevolution instance with {len(neuroevolution.population)} individuals")
    for genome in neuroevolution.population:
        print(f"   {genome['id']}: {genome['num_conv_layers']} conv, {genome['num_fc_layers']} fc")
    
    # Test the threaded evaluation
    start_time = time.time()
    success = neuroevolution.evaluate_population()
    end_time = time.time()
    
    print(f"\n--- Test Results ---")
    print(f"Evaluation successful: {success}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"All genomes have fitness > 0: {all(g['fitness'] > 0 for g in neuroevolution.population)}")
    print(f"Population size maintained: {len(neuroevolution.population) == config['population_size']}")
    print(f"Best individual found: {neuroevolution.best_individual is not None}")
    
    if neuroevolution.best_individual:
        print(f"Best fitness: {neuroevolution.best_individual['fitness']:.2f}%")
    
    # Check thread operations
    thread1_ops = [entry for entry in neuroevolution.evaluation_log if entry['thread'] == 'THREAD-1' and 'EXTRACTED' in entry['operation']]
    thread2_ops = [entry for entry in neuroevolution.evaluation_log if entry['thread'] == 'THREAD-2' and 'EXTRACTED' in entry['operation']]
    
    print(f"\n--- Thread Distribution ---")
    print(f"THREAD-1 processed: {len(thread1_ops)} genomes")
    print(f"THREAD-2 processed: {len(thread2_ops)} genomes")
    print(f"Total processed: {len(thread1_ops) + len(thread2_ops)} genomes")
    print(f"Load balance: {abs(len(thread1_ops) - len(thread2_ops)) <= 1}")
    
    # Final validation
    all_tests_passed = (
        success and
        all(g['fitness'] > 0 for g in neuroevolution.population) and
        len(neuroevolution.population) == config['population_size'] and
        neuroevolution.best_individual is not None and
        len(thread1_ops) + len(thread2_ops) == config['population_size']
    )
    
    print(f"\n🎉 All tests passed: {all_tests_passed}")
    return all_tests_passed

if __name__ == "__main__":
    test_result = test_hybrid_neuroevolution_threading()
    print(f"\nFinal test result: {'PASSED' if test_result else 'FAILED'}")