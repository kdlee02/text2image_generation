import os
import sys
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple
import fal_client
from dotenv import load_dotenv

load_dotenv()

class FALExperimentGenerator:
    def __init__(self, api_key: Optional[str] = None):

        if api_key:
            os.environ['FAL_KEY'] = api_key
        elif not os.getenv('FAL_KEY'):
            raise ValueError("API key required. Set FAL_KEY environment variable or provide api_key parameter.")
        
        self.images_dir = Path("generated_images")
        self.images_dir.mkdir(exist_ok=True)
    
    def generate_image(self, prompt: str, model: str) -> Tuple[str, Dict]:

        # minimal parameters (adjust if needed)
        arguments = {
            "prompt": prompt,
            "num_images": 1,
        }
        
        start_time = time.time()
        try:
            print(f"🎨 Generating image with {model}...")
            response = fal_client.run(model, arguments=arguments)
            end_time = time.time()
            time_taken = end_time - start_time
            
            if response and "images" in response and len(response["images"]) > 0:
                image_data = response["images"][0]
                metadata = {
                    "model": model,
                    "prompt": prompt,
                    "arguments": arguments,
                    "timestamp": datetime.now().isoformat(),
                    "time_taken": time_taken,
                    "response": response
                }
                print(f"✅ Image generated successfully in {time_taken:.2f}s")
                return image_data["url"], metadata
            else:
                raise Exception("No image generated in response")
                
        except Exception as e:
            end_time = time.time()
            time_taken = end_time - start_time
            raise Exception(f"Failed to generate image (took {time_taken:.2f}s): {str(e)}")
    
    def experiment_with_models(self, prompt: str, models: list, save_dir: Optional[str] = None) -> Dict[str, Dict]:

        results = {}
        
        if save_dir:
            save_path = self.images_dir / save_dir
            save_path.mkdir(exist_ok=True)
        else:
            save_path = self.images_dir
        
        print(f"\n🧪 Experimenting with {len(models)} models...")
        print(f"📝 Prompt: '{prompt}'")
        print("=" * 60)
        
        for i, model in enumerate(models, 1):
            model_name = model.split('/')[1:3] if '/' in model else model
            model_name = '-'.join(model_name)
            print(f"\n[{i}/{len(models)}] Testing {model_name}...")
            
            try:
                image_url, metadata = self.generate_image(prompt, model)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{model_name}_{timestamp}.jpg"
                filepath = save_path / filename
                
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                metadata["local_file"] = str(filepath)
                print(f"✅ {model_name}: Generated and saved to {filepath} (took {metadata['time_taken']:.2f}s)")
                print(f"🔗 URL: {image_url}")
                
                results[model] = {
                    "success": True,
                    "url": image_url,
                    "filepath": str(filepath),
                    "metadata": metadata
                }
                
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
                results[model] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results

def main():
    """Main function to run the multi-model experiment generator."""
    print("🧪 FAL AI Multi-Model Experiment Generator")
    print("=" * 50)
    
    api_key = os.getenv('FAL_KEY')
    if not api_key:
        api_key = input("Enter your FAL AI API key: ").strip()
        if not api_key:
            print("❌ Error: API key is required!")
            sys.exit(1)
    
    try:
        generator = FALExperimentGenerator(api_key)
        print("✅ FAL AI client initialized successfully!")
        
        models = {
            "1": ("fal-ai/imagen4/preview", "Imagen4 Ultra ($0.03 per image)"),
            "2": ("fal-ai/bytedance/dreamina/v3.1/text-to-image", "Dreamina v3.1 ($0.03 per image)"),
            "3": ("fal-ai/bytedance/seedream/v4/text-to-image", "Seedream v4"),
            "4": ("fal-ai/flux-pro/v1.1-ultra", "FLUX Pro v1.1 Ultra"),
            "5": ("fal-ai/ideogram/v3", "Ideogram v3"),
        }
        
        while True:
            print("\n" + "=" * 50)
            print("Choose an option:")
            print("1. Run multi-model experiment")
            print("2. Show available models")
            print("3. Quit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "3" or choice.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif choice == "1":

                prompt = input("\nEnter your image prompt: ").strip()
                if not prompt:
                    print("❌ Please enter a valid prompt.")
                    continue
                
                # Show models
                print("\nAvailable models:")
                for key, (endpoint, description) in models.items():
                    print(f"{key}. {description}")
                
                print(f"\nOptions:")
                print("- Enter model numbers (e.g., 1,3,5)")
                print("- Enter 'all' for all models")
                
                selection = input("\nSelect models: ").strip()
                
                if selection.lower() == 'all':
                    selected_models = [endpoint for endpoint, _ in models.values()]
                else:
                    try:
                        indices = [x.strip() for x in selection.split(',') if x.strip()]
                        selected_models = []
                        for idx in indices:
                            if idx in models:
                                selected_models.append(models[idx][0])
                            else:
                                print(f"⚠️  Invalid model number: {idx}")
                    except:
                        print("❌ Invalid selection format.")
                        continue
                
                if not selected_models:
                    print("❌ No valid models selected.")
                    continue
                
                save_dir = input("\nEnter directory name to save images (optional): ").strip()
                if not save_dir:
                    save_dir = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                try:
                    results = generator.experiment_with_models(prompt, selected_models, save_dir)
                    
                    print(f"\n📊 Experiment Summary:")
                    print("=" * 40)
                    successful = sum(1 for r in results.values() if r.get('success', False))
                    print(f"✅ Successful: {successful}/{len(results)}")
                    print(f"📁 Images saved to: generated_images/{save_dir}/")
                    
                    for model, result in results.items():
                        model_name = model.split('/')[-1] if '/' in model else model
                        status = "✅" if result.get('success', False) else "❌"
                        print(f"{status} {model_name}")
                
                except Exception as e:
                    print(f"❌ Experiment error: {str(e)}")
            
            elif choice == "2":
                print("\nAvailable models:")
                for key, (endpoint, description) in models.items():
                    print(f"{key}. {description}")
                    print(f"   Endpoint: {endpoint}")
            
            else:
                print("❌ Invalid choice. Please enter 1-3.")
    
    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()