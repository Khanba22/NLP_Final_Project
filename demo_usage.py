#!/usr/bin/env python3
"""
Demo script showing how to use the Technical Product Summarizer
"""

import requests
import json
from pprint import pprint

# Base URL for the API
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60 + "\n")

def demo_technical_summary():
    """Demo: Technical product summarization"""
    print_section("1. Technical Product Summarization")
    
    product_description = """
    The TechPro UltraBook X1 is a premium business laptop designed for professionals 
    who demand performance and portability. It features a 14-inch 4K OLED display 
    with 100% DCI-P3 color gamut. At its heart lies the latest Intel Core i7-13700H 
    processor with 14 cores and 20 threads, paired with 32GB of LPDDR5 RAM running 
    at 5200MHz. Storage is handled by a 1TB PCIe 4.0 NVMe SSD. The discrete NVIDIA 
    GeForce RTX 4060 graphics card with 8GB GDDR6 memory handles demanding applications. 
    The laptop weighs just 1.3kg and measures 15.9mm at its thickest point. Battery 
    life is impressive at up to 12 hours of mixed usage, with fast charging reaching 
    80% in 60 minutes. Connectivity includes Thunderbolt 4, USB-A 3.2, HDMI 2.1, and 
    Wi-Fi 6E. The aluminum unibody construction feels premium and durable.
    """
    
    response = requests.post(
        f"{BASE_URL}/technical-summarize",
        data={"text": product_description}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Product: {result['product_name']}")
        print(f"Category: {result['category']}")
        print(f"Price Range: {result['price_range']}")
        print(f"\nSummary:\n{result['summary']}")
        print(f"\nKey Specifications:")
        for spec, value in result['key_specs'].items():
            print(f"  - {spec}: {value}")
        print(f"\nPros:")
        for pro in result['pros']:
            print(f"  ✓ {pro}")
        print(f"\nCons:")
        for con in result['cons']:
            print(f"  ✗ {con}")
        print(f"\nBest For: {result['best_for']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

def demo_product_comparison():
    """Demo: Compare multiple products"""
    print_section("2. Product Comparison")
    
    product1 = """
    The TechPro UltraBook X1 features Intel Core i7-13700H, 32GB RAM, 1TB SSD, 
    RTX 4060 graphics, 14-inch 4K OLED display, weighs 1.3kg, 12-hour battery life. 
    Premium build quality with aluminum chassis. Excellent for professionals and 
    content creators. Price: $1999.
    """
    
    product2 = """
    The BudgetTech Essential 15 has AMD Ryzen 5 5500U processor, 8GB RAM, 512GB SSD,
    15.6-inch Full HD display, integrated graphics, weighs 1.8kg, 7-hour battery life.
    Plastic build but sturdy. Good for students and everyday use. Price: $549.
    """
    
    response = requests.post(
        f"{BASE_URL}/compare-products",
        data={"texts": [product1, product2]}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Comparing {result['product_count']} products")
        print(f"Category: {result['category']}")
        print(f"\nSummary: {result['summary']}")
        print(f"\nProducts:")
        for product in result['products']:
            print(f"\n  {product['name']}")
            print(f"    Category: {product['category']}")
            print(f"    Price: {product['price_range']}")
            print(f"    Best for: {product['best_for']}")
        
        if result['spec_comparison']:
            print(f"\nSpecification Comparison:")
            for spec_name, values in result['spec_comparison'].items():
                print(f"\n  {spec_name.upper()}:")
                for v in values:
                    print(f"    {v['product']}: {v['value']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

def demo_evaluation():
    """Demo: Evaluate summary quality"""
    print_section("3. Evaluation Metrics")
    
    reference = """
    The TechPro UltraBook X1 is a high-performance laptop with Intel Core i7 processor,
    32GB RAM, and 1TB SSD. It features a 4K OLED display and RTX 4060 graphics.
    The laptop is lightweight at 1.3kg with excellent 12-hour battery life.
    """
    
    generated = """
    The UltraBook X1 offers strong performance with Core i7 CPU, 32GB memory,
    and fast 1TB storage. It has a premium 4K screen and powerful RTX graphics.
    Very portable at 1.3kg and long-lasting battery up to 12 hours.
    """
    
    response = requests.post(
        f"{BASE_URL}/evaluate",
        data={
            "reference_text": reference,
            "generated_text": generated
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        scores = result['scores']
        print("ROUGE Scores:")
        print(f"  ROUGE-1 F1: {scores['rouge1_fmeasure']:.4f}")
        print(f"  ROUGE-2 F1: {scores['rouge2_fmeasure']:.4f}")
        print(f"  ROUGE-L F1: {scores['rougeL_fmeasure']:.4f}")
        print("\nBLEU Scores:")
        print(f"  BLEU-1: {scores['bleu1']:.4f}")
        print(f"  BLEU-2: {scores['bleu2']:.4f}")
        print(f"  BLEU-4: {scores['bleu4']:.4f}")
        print(f"\nOverall Score: {scores['overall_score']:.4f}")
        print(f"\nQuality: {scores['overall_score']*100:.1f}%")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

def demo_dataset_info():
    """Demo: Get dataset information"""
    print_section("4. Dataset Information")
    
    response = requests.get(f"{BASE_URL}/dataset-info")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total Products: {result['total_products']}")
        print(f"Categories: {', '.join(result['categories'])}")
        print("\nProducts in Dataset:")
        for product in result['products']:
            print(f"  - {product['name']} ({product['category']})")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

def demo_training():
    """Demo: Initiate model training"""
    print_section("5. Model Training")
    
    print("Note: Training is computationally intensive and time-consuming.")
    print("This demo initiates training with minimal configuration.\n")
    
    response = requests.post(
        f"{BASE_URL}/train",
        json={
            "num_epochs": 1,
            "batch_size": 2,
            "model_name": "facebook/bart-large-cnn"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        if result.get('metrics'):
            print("\nTraining Configuration:")
            for key, value in result['metrics'].items():
                print(f"  {key}: {value}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

def main():
    """Run all demos"""
    print("\n" + "="*60)
    print(" Technical Product Summarizer - Demo")
    print("="*60)
    print("\nMake sure the server is running:")
    print("  uvicorn app.main:app --reload")
    print("\nPress Enter to start demos...")
    input()
    
    try:
        # Check if server is running
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server is not running!")
            return
        print("✓ Server is running\n")
        
        # Run demos
        demo_technical_summary()
        input("\nPress Enter for next demo...")
        
        demo_product_comparison()
        input("\nPress Enter for next demo...")
        
        demo_evaluation()
        input("\nPress Enter for next demo...")
        
        demo_dataset_info()
        input("\nPress Enter for next demo...")
        
        demo_training()
        
        print("\n" + "="*60)
        print(" Demo Complete!")
        print("="*60)
        print("\nOpen http://localhost:8000 in your browser to use the UI.")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to server.")
        print("Make sure the server is running with:")
        print("  uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

