import os

import gradio as gr

# get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# check if the bias_options.txt file exists, if not, create it
bias_file = os.path.join(current_dir, "bias_options.txt")

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
    return string

def output_modifier(string):
    """
    This function is applied to the model outputs.
    """
    return string

def bot_prefix_modifier(string, params):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """
    if params['activate']:
        if params['use custom string']:
            return f'{string} {params["custom string"].strip()} '
        else:
            return f'{string} {params["bias string"].strip()} '
    else:
        return string

def ui():
    # Gradio elements
    activate = gr.Checkbox(value=params['activate'], label='Activate character bias')
    dropdown_string = gr.Dropdown(choices=bias_options, value=params["bias string"], label='Character bias', info='To edit the options in this dropdown edit the "bias_options.txt" file')
    use_custom_string = gr.Checkbox(value=False, label='Use custom bias textbox instead of dropdown', interactive=False)
    custom_string = gr.Textbox(value="", placeholder="Enter custom bias string", label="Custom Character Bias", info='To use this textbox activate the checkbox above')
    reset_button = gr.Button(value="Reset")

    def update_bias_string(x):
        if x:
            params.update({"bias string": x})
        else:
            params.update({"bias string": dropdown_string.get()})
        return x

    def update_custom_string(x):
        if x:
            params.update({"custom string": x.strip()})

    def reset_params():
        params.update({
            "activate": True,
            "bias string": "*I am so happy*",
            "use custom string": False,
            "custom string": ""
        })

    dropdown_string.change(update_bias_string, dropdown_string, None)
    custom_string.change(update_custom_string, custom_string, None)
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    use_custom_string.change(lambda x: params.update({"use custom string": x}), use_custom_string, None)
    reset_button.click(reset_params, None, None)

    # Group elements together depending on the selected option
    def bias_string_group():
        if use_custom_string.value:
            return gr.Group([use_custom_string, custom_string])
        else:
            return dropdown_string

    return gr.Interface(fn=bot_prefix_modifier, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Type something here...'), 
                         outputs='text', 
                         input_modifier=input_modifier, 
                         output_modifier=output_modifier, 
                         allow_flags=True, 
                         flags={"params": params}, 
                         title="Character Bias", 
                         description="This demo shows how to bias the behavior of a model using a character bias.", 
                         article="This demo uses a simple prefix modification technique to bias the behavior of a model. The prefix text is modified based on the selected character bias, which can be either selected from a dropdown or entered manually. The `bot_prefix_modifier` function is responsible for modifying the prefix text.", 
                         examples = [
                             {"input": "Tell me a story", "output": "Once upon a time, there was a happy little girl who lived in a happy little village. She was so happy that she spread happiness wherever she went
