import os

# Resolved from this file's location, not the process working directory.
# Previously this was the relative path 'data/image_prompt_template.txt', so running
# the app from anywhere but app/ silently fell back to a leftover "beautiful banner"
# prompt from an abandoned product: wrong prompt, HTTP 200, no signal.
TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'image_prompt_template.txt',
)


class PromptManager:
    """Manages the loading and formatting of the prompt template for consistency."""

    @classmethod
    def load_base_prompt(cls, file_path: str = TEMPLATE_PATH) -> str:
        """Load the base prompt from a file.

        Deliberately raises rather than substituting a different prompt. Generating
        against the wrong prompt silently is worse than failing.
        """
        with open(file_path, 'r') as f:
            return f.read()

    @classmethod
    def format_prompt(cls, user_input: str) -> str:
        """Format the prompt by inserting the user input into the template."""
        base_prompt = cls.load_base_prompt()
        return base_prompt.format(user_input=user_input)
