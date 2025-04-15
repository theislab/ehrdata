import ipywidgets as widgets


def create_single_option_widget(title_text: str, options: list[str]):
    """Create a single-option selection widget (RadioButtons).

    Args:
        title_text: The title displayed above the radio buttons.
        options: List of strings representing the radio button options.
        default_value: The default selected value.

    Returns:
        tuple: A VBox containing the UI elements and the RadioButtons widget.
    """
    title = widgets.HTML(f"<h3>{title_text}</h3>")
    default_value = options[0]
    radio_buttons = widgets.RadioButtons(options=options, value=default_value, layout=widgets.Layout(width="auto"))
    ui = widgets.VBox([title, radio_buttons])

    return ui, radio_buttons


def create_multiple_options_widget(title_text: str, options: list[str]):
    """Create a multiple-option selection widget (SelectMultiple).

    Args:
        title_text: The title displayed above the selection box.
        options: List of strings representing the options.

    Returns:
        tuple: A VBox containing the UI elements and the SelectMultiple widget.
    """
    title = widgets.HTML(f"<h3 style='color: #333; font-family: Arial, sans-serif;'>{title_text}</h3>")

    height = f"{min(30 * len(options), 300)}px"

    select_multiple = widgets.SelectMultiple(
        options=options, value=(options[0],), layout=widgets.Layout(width="100%", height=height, font_family="Arial")
    )

    ui = widgets.VBox(
        [title, select_multiple],
        layout=widgets.Layout(padding="10px", border="1px solid #ddd", border_radius="5px", width="50%"),
    )

    return ui, select_multiple
