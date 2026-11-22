import dspy
import os
import sys
from PIL import Image
from io import BytesIO
import requests
import fal_client
from dotenv import load_dotenv
import base64
import requests
from models.image_manager import ImageManager
from models.prompt_manager import PromptManager
load_dotenv()

# Every network read gets a deadline. Without one, a hung fal CDN connection
# blocks the worker thread forever.
REQUEST_TIMEOUT = 60


def _download_as_base64_jpeg(url):
    """Fetch an image URL and return it as base64 JPEG."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    # fal returns PNG; saving RGBA (logos, icons) as JPEG raises OSError.
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(APP_DIR)
IMAGES_LOG = os.path.join(APP_DIR, "imagesdata", "images_log.csv")

# Built once per process from the log as it stands at first use. New rows written
# during this run are not indexed until restart -- see README.
_retriever = None


def _retrieve_few_shot_examples(query, k=3):
    """Prompts that previously scored well, as guidance for the evaluator.

    Best-effort by design: a missing Upstage key, an unbuilt index or an empty
    corpus must degrade to no examples, never break image generation.
    """
    global _retriever
    try:
        if _retriever is None:
            if REPO_ROOT not in sys.path:
                sys.path.insert(0, REPO_ROOT)
            from rag.document_processor import DocumentProcessor

            _retriever = DocumentProcessor()
            _retriever.process_document(IMAGES_LOG)
        prompts = _retriever.retrieve_prompts(query, k=k)
        if not prompts:
            return ""
        return "\n\n".join(f"Example {i}: {p}" for i, p in enumerate(prompts, 1))
    except Exception as e:
        print(f"[rag] few-shot retrieval unavailable ({e}); continuing without examples")
        return ""


def _score(result):
    """overall_score is LLM output and may not parse as an int."""
    try:
        return int(float(result.overall_score))
    except (TypeError, ValueError, AttributeError):
        return -1

class ImageEvaluationSignature(dspy.Signature):
    """
    You are an expert art critic and image generation specialist tasked with evaluating generated images against their intended prompts.

    YOUR EVALUATION PROCESS:
    1. Carefully examine the generated image and compare it to the desired_prompt
    2. Break down your analysis into distinct components: subject, art type, art style, and art movement
    3. Review previous_attempts to avoid repeating unsuccessful approaches, and few_shot_examples for how effective prompts are phrased
    4. Identify any conflicting or contradictory visual elements
    5. Provide specific, actionable feedback for each component
    6. Generate a revised prompt that directly addresses identified issues

    Provide thorough reasoning before making judgments. Be honest and constructive in your evaluation.
    """
    
    desired_prompt: str = dspy.InputField(desc="The original prompt the user wanted")
    current_image: dspy.Image = dspy.InputField(desc="The generated image to evaluate")
    current_prompt: str = dspy.InputField(desc="The prompt used to generate this image")
    previous_attempts: str = dspy.InputField(desc="History of previous attempts and feedback")
    few_shot_examples: str = dspy.InputField(desc="Prompts that previously scored highly after revision. Use them as guidance for how to phrase an effective revised_prompt, not as content to copy. May be empty.")
    
    reasoning: str = dspy.OutputField(desc="Step-by-step analysis of the image")
    overall_prompt_match: bool = dspy.OutputField(desc="Does the image match the overall intent?")
    subject_match: bool = dspy.OutputField(desc="Is the main subject correct?")
    art_type_match: bool = dspy.OutputField(desc="Is the art type/medium correct?")
    art_style_match: bool = dspy.OutputField(desc="Is the artistic style correct?")
    art_movement_match: bool = dspy.OutputField(desc="Is the art movement/period correct?")
    has_conflicting_elements: bool = dspy.OutputField(desc="Are there contradictory elements?")
    
    conflict_description: str = dspy.OutputField(desc="Describe any conflicting elements found")
    overall_prompt_match_feedback: str = dspy.OutputField(desc="Feedback on overall match")
    subject_feedback: str = dspy.OutputField(desc="Specific feedback on the subject")
    art_type_feedback: str = dspy.OutputField(desc="Specific feedback on art type")
    art_style_feedback: str = dspy.OutputField(desc="Specific feedback on art style")
    art_movement_feedback: str = dspy.OutputField(desc="Specific feedback on art movement")
    revised_prompt: str = dspy.OutputField(desc="Improved prompt addressing identified issues")
    overall_score: int = dspy.OutputField(desc="Overall Score between 1 to 10 where 10 means perfect")
class ImageGeneratorService:
    def __init__(self):
        self.image_manager = ImageManager()
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY environment variable is required for LLM")
        if not os.getenv('FAL_KEY'):
            raise ValueError("FAL_KEY environment variable is required for Fal AI")
        
        # Configure fal_client with API key
        os.environ['FAL_KEY'] = os.getenv('FAL_KEY')
        
        lm = dspy.LM(model="gemini/gemini-2.5-pro",api_key=os.environ["GEMINI_API_KEY"])
        dspy.settings.configure(lm=lm)

    def generate_image(self, prompt):
        request_id = fal_client.submit(
            "fal-ai/imagen4/preview",
            arguments={
                "prompt": prompt
            },
        ).request_id

        result = fal_client.result("fal-ai/imagen4/preview", request_id)
        url = result["images"][0]["url"]
        # Don't call add_image here - return image_data for later logging
        return dspy.Image.from_url(url), url, result["images"][0]

    def dspy_opt(self, user_input):
        # Keep the raw input: it is what gets logged as original_user_prompt and
        # what the retriever should match on. This used to be rebound to the
        # formatted prompt, so the log recorded the template output as the "original".
        original_user_prompt = user_input
        initial_prompt = PromptManager.format_prompt(user_input)
        current_prompt = initial_prompt

        few_shot_examples = _retrieve_few_shot_examples(original_user_prompt)

        check_and_revise_prompt = dspy.ChainOfThought(ImageEvaluationSignature)

        history = []

        url = ''
        # The loop used to return whichever image came last. Measured over 74 logged
        # runs, that returned a worse image than one already generated in 11 of the
        # 15 runs that exhausted all 5 iterations.
        best_score = float('-inf')
        best_url = ''

        for i in range(5):
            current_image, url, image_data = self.generate_image(current_prompt)
            #display_image(current_image)
            #print(f'current prompt: {current_prompt}')
            # Format only the last attempt for context (more focused feedback)
            if history:
                last_attempt = history[-1]
                history_str = (
                    f"Previous attempt:\n"
                    f"Prompt: {last_attempt['prompt']}\n"
                    f"Subject: {last_attempt['detailed_feedback']['subject']}\n"
                    f"Art Type: {last_attempt['detailed_feedback']['art_type']}\n"
                    f"Style: {last_attempt['detailed_feedback']['style']}\n"
                    f"Art Movement: {last_attempt['detailed_feedback']['art_movement']}\n"
                    f"Conflicts: {last_attempt['detailed_feedback']['conflicts']}"
                )
            else:
                history_str = "No previous attempts"

            result = check_and_revise_prompt(
                desired_prompt=initial_prompt, 
                current_image=current_image, 
                current_prompt=current_prompt,
                previous_attempts=history_str,
                few_shot_examples=few_shot_examples
            )

            score = _score(result)
            if score > best_score:
                best_score, best_url = score, url
            
            # Log image with DSPy evaluation data
            self.image_manager.add_image(current_prompt, image_data, result, original_user_prompt)
            # Store this attempt
            history.append({
                'prompt': current_prompt,
                'detailed_feedback': {
                'subject': result.subject_feedback,
                'art_type': result.art_type_feedback,
                'style': result.art_style_feedback,
                'art_movement': result.art_movement_feedback,
                'conflicts': result.conflict_description
            }
            })
            
            if result.subject_match and result.art_type_match and result.art_style_match and result.overall_prompt_match and not result.has_conflicting_elements and result.art_movement_match:
                print("")
                print("✓ All components match perfectly!")
                print(f"Subject - {result.subject_feedback}")
                print(f"Art Type - {result.art_type_feedback}")
                print(f"Art Style - {result.art_style_feedback}")
                print(f"Art Movement - {result.art_movement_feedback}")
                print(f"Overall Prompt - {result.overall_prompt_match_feedback}")
                print(f"Confliction - {result.conflict_description}")
                print(f"Overall Score: {result.overall_score}")
                return _download_as_base64_jpeg(url)
            else:
                print(f"Subject Match: {'✓' if result.subject_match else '✗'} - {result.subject_feedback}")
                print(f"Art Type Match: {'✓' if result.art_type_match else '✗'} - {result.art_type_feedback}")
                print(f"Art Movement Match: {'✓' if result.art_movement_match else '✗'} - {result.art_movement_feedback}")
                print(f"Art Style Match: {'✓' if result.art_style_match else '✗'} - {result.art_style_feedback}")
                print(f"Overall Prompt Match: {'✓' if result.overall_prompt_match else '✗'} - {result.overall_prompt_match_feedback}")
                print(f"Confliction: {'✓' if not result.has_conflicting_elements else '✗'} - {result.conflict_description}")
                print(f"\nRevised prompt: {result.revised_prompt}")
                print(f"Overall Score: {result.overall_score}")
                current_prompt = result.revised_prompt

        print(f"\nNo iteration passed every check. "
              f"Returning the best-scoring attempt (score {best_score} of 10).")
        return _download_as_base64_jpeg(best_url or url)
    
    def get_all_images(self):
        return self.image_manager.get_images()
    
    def get_csv_log(self):
        return self.image_manager.get_csv_log()