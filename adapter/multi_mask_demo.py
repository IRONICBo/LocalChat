import gradio as gr
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Initialize the analyzer and anonymizer engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def mask_sensitive_info(content: str) -> str:
    """Identify and mask sensitive information in content with numbered entities"""
    # Analyze the text to find PII entities
    results = analyzer.analyze(text=content, language="en")

    # Prepare anonymization operations
    operations = []
    for result in results:
        # Create operator config to replace sensitive data with a placeholder
        operations.append(
            OperatorConfig("replace", {"new_value": f"[{result.entity_type}]"})
        )

    # Anonymize with custom operations
    anonymized_result = anonymizer.anonymize(
        text=content,
        analyzer_results=results,
        operators={
            result.entity_type: operations[i] for i, result in enumerate(results)
        },
    )

    return anonymized_result.text

def mask_tab():
    with gr.Tab(label="Mask Sensitive Information"):
        gr.Markdown("Enter a prompt to mask sensitive information.")

        with gr.Row():
            input_text = gr.Textbox(label="Input Prompt", lines=10)
            output_text = gr.Textbox(label="Masked Prompt", lines=10, interactive=False)

        submit_button = gr.Button("Submit")

        def update_masked_prompt(prompt):
            masked_prompt = mask_sensitive_info(prompt)
            return masked_prompt

        submit_button.click(fn=update_masked_prompt, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>Masking Manager</center></h1>")
        mask_tab()

    main_block.queue()
    main_block.launch()