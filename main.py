"""
Main execution script for the Denoising Project.
"""
from src.sdk.sdk import SineDenoisingSDK

if __name__ == "__main__":
    print("Initializing Sine Denoising System...")
    
    # Instantiate the SDK
    sdk = SineDenoisingSDK()
    
    # Step 1: Generate Data
    sdk.initialize_dataset()
    
    # Step 2: Train the models
    sdk.train_models(epochs=50)
    
    print("\nTraining complete!")